use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering}
    },
    time::{Instant, SystemTime},
};

use crossbeam_channel::{bounded, Receiver, Sender, TryRecvError};
use eframe::{egui, egui::TextureHandle, App};
use egui::Vec2;

use crate::load::{self, ImgMsg, JobMsg, PathMsg};
use crate::sort;

/* ───────────────────────── UI tuneables ─────────────────────────── */

const UPLOADS_PER_FRAME: usize = 4;
const PATHS_PER_FRAME: usize   = 64;

/* ───────────────────────── domain types ─────────────────────────── */

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum SortKey { Name, Created, Size, Height, Width, Type }
impl SortKey {
    pub fn all() -> [Self; 6] { [Self::Name, Self::Created, Self::Size, Self::Height, Self::Width, Self::Type] }
    pub fn label(self) -> &'static str {
        match self {
            Self::Name => "Name", Self::Created => "Date", Self::Size => "Size",
            Self::Height => "Height", Self::Width => "Width", Self::Type => "Type",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Layout { One, Two, Three }
impl Layout {
    fn all() -> [Self; 3] { [Self::One, Self::Two, Self::Three] }
    fn label(self) -> &'static str {
        match self { Self::One => "1 image", Self::Two => "2 images", Self::Three => "3 images" }
    }
}

/* Loaded image entry (has GPU texture) */
pub(crate) struct ImgEntry {
    pub(crate) path: PathBuf,
    pub(crate) name: String,
    pub(crate) ext: String,
    pub(crate) tex: TextureHandle,
    pub(crate) w: u32,
    pub(crate) h: u32,
    pub(crate) bytes: u64,
    pub(crate) created: Option<SystemTime>,
}

/* Discovered file metadata (lightweight; no decode) */
#[derive(Clone)]
pub(crate) struct FileMeta {
    pub(crate) path: PathBuf,
    pub(crate) name: String,
    pub(crate) ext: String,
    pub(crate) bytes: u64,
    pub(crate) created: Option<SystemTime>,
}

/* ───────────────────────── app state ─────────────────────────────── */

pub struct ViewerApp {
    images: Vec<ImgEntry>,
    files: Vec<FileMeta>,
    index_of: HashMap<PathBuf, usize>,

    current: usize,

    img_rx: Receiver<ImgMsg>,
    img_tx: Sender<ImgMsg>,

    paths_rx: Receiver<PathMsg>,
    paths_tx: Sender<PathMsg>,

    // Normal jobs + a small priority queue for immediate targets
    job_tx: Sender<JobMsg>,
    job_rx: Receiver<JobMsg>,
    job_tx_prio: Sender<JobMsg>,
    job_rx_prio: Receiver<JobMsg>,

    // IO/prefetch state (GUI-agnostic types from load.rs)
    qstate: load::QueueState,
    prefetch: load::Prefetcher,

    // generation token: bump on every new load
    current_gen: Arc<AtomicU64>,

    current_dir: Option<PathBuf>,

    // We manage *borderless* fullscreen ourselves (not OS fullscreen):
    is_borderless_fs: bool,
    prev_win_pos: Option<egui::Pos2>,
    prev_win_size: Option<egui::Vec2>,

    sort_key: SortKey,
    ascending: bool,
    layout: Layout,

    // UI chrome (both top and bottom panels share this single flag)
    show_top_bar: bool,
    prev_alt_down: bool,

    // Remember previous chrome visibility when entering fullscreen
    prev_chrome_visible: Option<bool>,

    zoom: f32,
    pan: Vec2,
    last_cursor: Option<egui::Pos2>,
    pending_target: Option<String>,

    egui_ctx: egui::Context,

    // For ultra-fast double-click detection
    last_primary_down: Option<Instant>,
}

impl ViewerApp {
    pub fn new(egui_ctx: egui::Context) -> Self {
        let (img_tx, img_rx)     = bounded::<ImgMsg>(load::IMG_CHAN_CAP);
        let (paths_tx, paths_rx) = bounded::<PathMsg>(load::PATHS_CHAN_CAP);
        let (job_tx, job_rx)     = bounded::<JobMsg>(load::MAX_ENQUEUED_JOBS);
        let (job_tx_prio, job_rx_prio) = bounded::<JobMsg>(256);

        Self {
            images: Vec::new(),
            files: Vec::new(),
            index_of: HashMap::new(),
            current: 0,
            img_rx, img_tx,
            paths_rx, paths_tx,
            job_tx, job_rx,
            job_tx_prio, job_rx_prio,

            qstate: load::QueueState::default(),
            prefetch: load::Prefetcher::default(),
            current_gen: Arc::new(AtomicU64::new(1)),

            current_dir: None,

            is_borderless_fs: false,
            prev_win_pos: None,
            prev_win_size: None,

            sort_key: SortKey::Name,
            ascending: true,
            layout: Layout::One,

            show_top_bar: true,
            prev_alt_down: false,
            prev_chrome_visible: None,

            zoom: 1.0,
            pan: Vec2::ZERO,
            last_cursor: None,
            pending_target: None,

            egui_ctx,
            last_primary_down: None,
        }
    }

    fn launch_workers(&self, use_downscaled: bool, max_dim: u32) {
        // Dedicated decoder pool; keep walkers separate (in load::enumerate_paths)
        let threads = {
            const MIN: usize = 2; const MAX: usize = 12;
            let logical = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
            ((logical * 2) / 3).clamp(MIN, MAX)
        };
        load::start_decoder_workers_pool(
            self.job_rx.clone(),
            Some(self.job_rx_prio.clone()),
            self.img_tx.clone(),
            self.egui_ctx.clone(),
            self.current_gen.clone(),
            threads,
            use_downscaled,
            max_dim,
        );
    }

    pub fn spawn_loader(&mut self, dir: PathBuf) {
        self.reset_for_new_load();
        self.current_dir = Some(dir.clone());
        self.launch_workers(true, 2560);
        let paths_tx = self.paths_tx.clone();
        let ctx      = self.egui_ctx.clone();
        std::thread::spawn(move || load::enumerate_paths(dir, paths_tx, ctx));
    }

    pub fn spawn_loader_file(&mut self, file: PathBuf) {
        self.reset_for_new_load();
        self.launch_workers(true, 2560);
        self.pending_target = file.file_name().and_then(|s| s.to_str()).map(|s| s.to_owned());

        let gen = self.current_gen.load(Ordering::Relaxed);

        if let Some(parent) = file.parent().map(|p| p.to_path_buf()) {
            self.current_dir = Some(parent.clone());

            // immediate decode for the target file (priority queue)
            let _ = self.job_tx_prio.try_send((gen, file.clone()));

            let paths_tx = self.paths_tx.clone();
            let ctx      = self.egui_ctx.clone();
            std::thread::spawn(move || load::enumerate_paths(parent, paths_tx, ctx));
        } else {
            self.current_dir = None;
            let _ = self.job_tx_prio.try_send((gen, file.clone()));
        }
        self.prefetch.dirty = true;
    }

    pub fn add_folder(&mut self, dir: PathBuf) {
        self.current_dir = Some(dir.clone());
        let paths_tx = self.paths_tx.clone();
        let ctx      = self.egui_ctx.clone();
        std::thread::spawn(move || load::enumerate_paths(dir, paths_tx, ctx));
        self.prefetch.dirty = true;
    }

    fn toggle_borderless_fullscreen(&mut self, ctx: &egui::Context) {
        let (cur_pos, cur_size, monitor_size) = ctx.input(|i| {
            let vp = &i.viewport();
            let cur_pos  = vp
                .outer_rect
                .unwrap_or(egui::Rect::from_min_size(egui::pos2(0.0,0.0), i.screen_rect().size()))
                .min;
            let cur_size = vp.inner_rect.unwrap_or(i.screen_rect()).size();
            let mon_size = vp.monitor_size.unwrap_or(i.screen_rect().size());
            (cur_pos, cur_size, mon_size)
        });

        // Reset/center the image every time we toggle
        self.zoom = 1.0;
        self.pan = Vec2::ZERO;
        self.last_cursor = None;

        if !self.is_borderless_fs {
            // Entering fullscreen: hide chrome, remember previous chrome state
            self.prev_chrome_visible = Some(self.show_top_bar);
            self.show_top_bar = false;

            self.prev_win_pos  = Some(cur_pos);
            self.prev_win_size = Some(cur_size);
            ctx.send_viewport_cmd(egui::ViewportCommand::Decorations(false));
            ctx.send_viewport_cmd(egui::ViewportCommand::OuterPosition(egui::pos2(0.0, 0.0)));
            ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(monitor_size));
            self.is_borderless_fs = true;
        } else {
            // Exiting fullscreen: restore chrome visibility if known, else default to true
            if let Some(prev) = self.prev_chrome_visible.take() {
                self.show_top_bar = prev;
            } else {
                self.show_top_bar = true;
            }

            ctx.send_viewport_cmd(egui::ViewportCommand::Decorations(true));
            if let Some(s) = self.prev_win_size { ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(s)); }
            if let Some(p) = self.prev_win_pos  { ctx.send_viewport_cmd(egui::ViewportCommand::OuterPosition(p)); }
            self.is_borderless_fs = false;
        }
        ctx.request_repaint();
    }

    fn current_img(&self) -> Option<&ImgEntry> { self.images.get(self.current) }

    fn delete_current(&mut self) {
        if self.images.is_empty() { return; }
        let path = self.images[self.current].path.clone();
        let _ = std::fs::remove_file(&path);
        self.images.remove(self.current);
        self.qstate.seen.remove(&path);
        if self.current >= self.images.len() && !self.images.is_empty() {
            self.current = self.images.len() - 1;
        }
        self.zoom = 1.; self.pan=Vec2::ZERO; self.last_cursor=None;
        self.prefetch.dirty = true;
    }

    fn reset_for_new_load(&mut self) {
        self.images.clear();
        self.files.clear();
        self.index_of.clear();
        self.current = 0;
        self.qstate.clear();
        self.pending_target = None;
        self.prefetch.reset();
        // bump generation so stale jobs are ignored by workers
        self.current_gen.fetch_add(1, Ordering::Relaxed);
        // also clear enqueued set to allow requeue under new ordering
        self.qstate.enqueued.clear();
    }

    fn next(&mut self) {
        if self.images.is_empty() { return; }
        self.current = (self.current+1)%self.images.len();
        self.zoom=1.;self.pan=Vec2::ZERO;self.last_cursor=None;
        self.prefetch.dirty = true;
    }
    fn prev(&mut self) {
        if self.images.is_empty() { return; }
        self.current = (self.current+self.images.len()-1)%self.images.len();
        self.zoom=1.;self.pan=Vec2::ZERO;self.last_cursor=None;
        self.prefetch.dirty = true;
    }

    fn recenter_prefetch(&mut self) {
        let paths: Vec<PathBuf> = self.files.iter().map(|m| m.path.clone()).collect();
        let cur_path_owned: Option<PathBuf> = self.current_img().map(|e| e.path.clone());
        let cur_path_ref: Option<&PathBuf> = cur_path_owned.as_ref();
        let pending = self.pending_target.as_deref();
        self.prefetch.recenter(&paths, cur_path_ref, pending, &self.index_of);
    }
}

/* ─────────────────── eframe integration ───────────────────────── */
impl App for ViewerApp {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        use egui::{Color32, ColorImage, CursorIcon, Key, Label, PointerButton, Pos2, Rect, Sense};

        fn no_sel(text: impl Into<egui::WidgetText>) -> Label { Label::new(text).selectable(false) }

        let input = ctx.input(|i| i.clone());

        // Ultra-fast global double-click anywhere (raw press pairing)
        {
            use std::time::Duration;
            const DC_WINDOW: Duration = Duration::from_millis(300);

            for ev in &input.events {
                if let egui::Event::PointerButton { button: egui::PointerButton::Primary, pressed: true, .. } = ev {
                    let now = Instant::now();
                    if let Some(t0) = self.last_primary_down {
                        if now.duration_since(t0) <= DC_WINDOW {
                            self.toggle_borderless_fullscreen(ctx);
                            self.last_primary_down = None; // consume the pair
                            continue;
                        }
                    }
                    self.last_primary_down = Some(now);
                }
            }
        }

        // Toggle chrome with Alt (applies to both top and bottom)
        if input.modifiers.alt && !self.prev_alt_down { self.show_top_bar = !self.show_top_bar; }
        self.prev_alt_down = input.modifiers.alt;

        // 1) Drain decoded images → upload textures (now includes metadata; no fs::metadata here)
        let mut uploaded = 0usize;
        loop {
            match self.img_rx.try_recv() {
                Ok((path, w, h, rgba, bytes, created)) => {
                    if self.qstate.seen.contains(&path) { continue; }

                    let tex = ctx.load_texture(
                        path.file_name().unwrap().to_string_lossy(),
                        ColorImage::from_rgba_unmultiplied([w, h], &rgba),
                        egui::TextureOptions::default(),
                    );

                    self.images.push(ImgEntry {
                        path: path.clone(),
                        name: tex.name().into(),
                        ext : tex.name().rsplit('.').next().unwrap_or("").into(),
                        tex, w: w as u32, h: h as u32, bytes, created,
                    });

                    self.qstate.mark_seen(&path);
                    uploaded += 1;
                    if uploaded >= UPLOADS_PER_FRAME { break; }
                }
                Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => break,
            }
        }
        if uploaded > 0 { ctx.request_repaint(); }

        // 2) Drain discovered paths → index & (re)sort files as needed
        let mut got_paths = false;
        for _ in 0..PATHS_PER_FRAME {
            match self.paths_rx.try_recv() {
                Ok(p) => {
                    let meta = lightweight_meta(&p);
                    self.index_of.insert(p.clone(), self.files.len());
                    self.files.push(meta);
                    got_paths = true;
                }
                Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => break,
            }
        }
        if got_paths {
            sort::sort_files_lightweight(&mut self.files, self.sort_key, self.ascending, &mut self.index_of);
            self.recenter_prefetch();
            ctx.request_repaint();
        }

        // 3) Prefetch scheduling (pure load-layer logic)
        {
            let paths: Vec<PathBuf> = self.files.iter().map(|m| m.path.clone()).collect();
            let gen = self.current_gen.load(Ordering::Relaxed);
            let _enq = self.prefetch.tick(
                &paths,
                &mut self.qstate,
                &self.job_tx,
                load::PREFETCH_PER_TICK,
                load::MAX_ENQUEUED_JOBS,
                gen,
            );
            if _enq > 0 { ctx.request_repaint(); }
        }

        // 4) Hotkeys
        if input.key_pressed(Key::ArrowRight) { self.next(); }
        if input.key_pressed(Key::ArrowLeft)  { self.prev(); }

        // Keyboard fullscreen toggle uses *borderless* fullscreen too
        if input.key_pressed(Key::F11) {
            self.toggle_borderless_fullscreen(ctx);
        }
        if input.key_pressed(Key::Escape) && self.is_borderless_fs {
            self.toggle_borderless_fullscreen(ctx);
        }

        if input.key_pressed(Key::Delete) { self.delete_current(); }
        match input.raw_scroll_delta.y {
            d if d > 0.0 => self.prev(),
            d if d < 0.0 => self.next(),
            _ => {}
        }

        // 5) Central panel (drag/pan via click_and_drag; double-click handled globally)
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.images.is_empty() {
                ui.centered_and_justified(|ui| ui.add(no_sel("Loading images…")));
                return;
            }

            let full  = ui.available_rect_before_wrap();
            let bar_h = if self.show_top_bar { 32.0 } else { 0.0 };
            let avail = Rect::from_min_max(Pos2::new(full.min.x, full.min.y + bar_h),
                                           Pos2::new(full.max.x, full.max.y - bar_h));

            if let Some(p) = input.pointer.hover_pos() { self.last_cursor = Some(p); }
            let cursor = self.last_cursor.unwrap_or(avail.center());

            // Mouse-side-button zoom
            if input.pointer.button_pressed(PointerButton::Extra2) {
                let nz = (self.zoom * 1.1).clamp(0.1, 10.0);
                self.pan += (cursor - (avail.center()+self.pan)) * (1.0 - nz/self.zoom);
                self.zoom = nz;
            }
            if input.pointer.button_pressed(PointerButton::Extra1) {
                let nz = (self.zoom / 1.1).clamp(0.1, 10.0);
                self.pan += (cursor - (avail.center()+self.pan)) * (1.0 - nz/self.zoom);
                self.zoom = nz;
            }

            let set_grab = |grabbing: bool| {
                ctx.output_mut(|o| {
                    o.cursor_icon = if grabbing { CursorIcon::Grabbing } else { CursorIcon::Grab };
                });
            };

            match self.layout {
                Layout::One => {
                    let img  = self.current_img().unwrap();
                    let base = img.tex.size_vec2(); // full texture size (native)
                    let fit  = (avail.width()/base.x).min(avail.height()/base.y).min(1.0);
                    let size = base * fit * self.zoom;
                    let rect = Rect::from_center_size(avail.center()+self.pan, size);

                    let resp = ui.allocate_rect(rect, Sense::click_and_drag());

                    ui.painter().image(
                        img.tex.id(), rect,
                        egui::Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0,1.0)),
                        Color32::WHITE,
                    );

                    if resp.hovered() { set_grab(false); }
                    if resp.dragged()  { self.pan += resp.drag_delta(); set_grab(true); }
                }
                Layout::Two | Layout::Three => {
                    let n = if self.layout == Layout::Two { 2 } else { 3 };
                    let mut tiles = Vec::with_capacity(n);
                    for i in 0..n {
                        let e   = &self.images[(self.current+i)%self.images.len()];
                        let fit = (avail.width()/(n as f32)/e.tex.size_vec2().x)
                                   .min(avail.height()/e.tex.size_vec2().y).min(1.0);
                        tiles.push((e.tex.id(), e.tex.size_vec2()*fit*self.zoom));
                    }
                    let w: f32 = tiles.iter().map(|(_,s)|s.x).sum();
                    let h: f32 = tiles.iter().map(|(_,s)|s.y).fold(0.0, f32::max);
                    let rect   = Rect::from_center_size(avail.center()+self.pan, Vec2::new(w,h));

                    let resp   = ui.allocate_rect(rect, Sense::click_and_drag());

                    ui.allocate_ui_at_rect(rect, |ui|{
                        ui.horizontal_centered(|ui| for (id,sz) in tiles { ui.image((id,sz)); });
                    });

                    if resp.hovered() { set_grab(false); }
                    if resp.dragged()  { self.pan += resp.drag_delta(); set_grab(true); }
                }
            }
        });

        // 6) Menu + stats (both are hidden when show_top_bar is false)
        if self.show_top_bar {
            egui::TopBottomPanel::top("menu").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Open file…").clicked() {
                        if let Some(f) = rfd::FileDialog::new()
                            .add_filter("Images",&["png","jpg","jpeg","bmp","gif","tiff","webp"])
                            .pick_file() { self.spawn_loader_file(f); }
                    }
                    if ui.button("Add folder…").clicked() {
                        if let Some(d) = rfd::FileDialog::new().pick_folder() { self.add_folder(d); }
                    }
                    ui.separator();

                    let prev = (self.layout, self.sort_key, self.ascending);
                    ui.label("View:");
                    egui::ComboBox::from_id_source("layout")
                        .selected_text(self.layout.label())
                        .show_ui(ui, |ui| for lay in Layout::all() {
                            ui.selectable_value(&mut self.layout, lay, lay.label());
                        });
                    ui.separator();
                    ui.label("Sort by:");
                    egui::ComboBox::from_id_source("sort_key")
                        .selected_text(self.sort_key.label())
                        .show_ui(ui, |ui| for k in SortKey::all() {
                            ui.selectable_value(&mut self.sort_key, k, k.label());
                        });
                    ui.label("Order:");
                    egui::ComboBox::from_id_source("sort_ord")
                        .selected_text(if self.ascending { "Asc" } else { "Desc" })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.ascending, true,  "Asc");
                            ui.selectable_value(&mut self.ascending, false, "Desc");
                        });

                    if prev != (self.layout, self.sort_key, self.ascending) {
                        sort::sort_images(&mut self.images, self.sort_key, self.ascending);
                        sort::sort_files_lightweight(&mut self.files, self.sort_key, self.ascending, &mut self.index_of);

                        self.zoom = 1.0; self.pan = Vec2::ZERO; self.last_cursor = None;

                        // Allow prefetch to repopulate according to new order without re-decoding seen items.
                        self.qstate.enqueued.clear();
                        self.recenter_prefetch();
                    }

                    ui.separator();
                    if ui.add_enabled(!self.images.is_empty(), egui::Button::new("◀ Prev")).clicked() { self.prev(); }
                    if ui.add_enabled(!self.images.is_empty(), egui::Button::new("Next ▶")).clicked() { self.next(); }
                });
            });

            egui::TopBottomPanel::bottom("stats").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if let Some(dir) = &self.current_dir {
                        ui.label(format!("Last folder: {}", dir.display())); ui.separator();
                    }
                    if let Some(img) = self.current_img() {
                        ui.label(&img.name); ui.separator();
                        ui.label(format!("{} / {}", self.current+1, self.images.len())); ui.separator();
                        ui.label(format!("{}×{}", img.w, img.h)); ui.separator();
                        ui.label(human_bytes(img.bytes));
                    }
                });
            });
        }

        // Default cursor if we didn't set Grab/Grabbing above:
        ctx.output_mut(|o| if !matches!(o.cursor_icon, CursorIcon::Grab|CursorIcon::Grabbing) {
            o.cursor_icon = CursorIcon::Default;
        });
    }
}

/* ───────────────────────── helpers ──────────────────────────── */

fn lightweight_meta(path: &PathBuf) -> FileMeta {
    use std::fs;
    let (bytes, created) = fs::metadata(path).map(|m| (m.len(), m.created().ok())).unwrap_or((0, None));
    let name = path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let ext  = path.extension().map(|s| s.to_string_lossy().to_lowercase()).unwrap_or_default();
    FileMeta { path: path.clone(), name, ext, bytes, created }
}

fn human_bytes(b: u64) -> String {
    let f = b as f64;
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    if f >= GB { format!("{:.1} GB", f / GB) }
    else if f >= MB { format!("{:.1} MB", f / MB) }
    else if f >= KB { format!("{:.1} KB", f / KB) }
    else { format!("{b} B") }
}
