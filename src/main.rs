#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{
    collections::{HashMap, HashSet},
    env,
    path::PathBuf,
    time::{SystemTime, Instant},
};

use crossbeam_channel::{bounded, Receiver, Sender, TryRecvError};
use eframe::{
    egui,
    egui::{TextureHandle, ViewportBuilder},
    NativeOptions,
};
use egui::Vec2;
use image::{DynamicImage, ImageReader};
use walkdir::WalkDir;

/* ───────────────────────── types / tuneables ─────────────────────── */

type ImgMsg  = (PathBuf, usize, usize, Vec<u8>); // (path,w,h,rgba)
type JobMsg  = PathBuf;                           // decode this path
type PathMsg = PathBuf;                           // discovered path

// Keep UI responsive by hard-capping per-frame work:
const UPLOADS_PER_FRAME: usize   = 4;
const PATHS_PER_FRAME: usize     = 64;
const PREFETCH_PER_TICK: usize   = 64;

// How far around current to prefetch in lightweight order
const PREFETCH_RADIUS: usize     = 512;

// Back-pressure for decode job queue
const MAX_ENQUEUED_JOBS: usize   = 2048;

/* ───────────────────────── program entry ─────────────────────────── */

fn main() -> eframe::Result<()> {
    let start_path = env::args().nth(1).map(PathBuf::from);

    let mut opts = NativeOptions::default();
    // Normal decorated window in windowed mode; we only go borderless while "fullscreen".
    opts.viewport = ViewportBuilder::default()
        .with_inner_size([960.0, 540.0])
        .with_decorations(true);

    eframe::run_native(
        "Rust Multicore Image Viewer",
        opts,
        Box::new(move |cc| {
            let mut app = ViewerApp::new(cc.egui_ctx.clone());
            if let Some(p) = &start_path {
                if p.is_file() { app.spawn_loader_file(p.clone()); }
                else if p.is_dir() { app.spawn_loader(p.clone()); }
            }
            Box::new(app)
        }),
    )
}

/* ───────────────────────── domain types ─────────────────────────── */

#[derive(Clone, Copy, PartialEq, Eq)]
enum SortKey { Name, Created, Size, Height, Width, Type }
impl SortKey {
    fn all() -> [Self; 6] { [Self::Name, Self::Created, Self::Size, Self::Height, Self::Width, Self::Type] }
    fn label(self) -> &'static str {
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
struct ImgEntry {
    path: PathBuf,
    name: String,
    ext: String,
    tex: TextureHandle,
    w: u32,
    h: u32,
    bytes: u64,
    created: Option<SystemTime>,
}

/* Discovered file metadata (lightweight; no decode) */
#[derive(Clone)]
struct FileMeta {
    path: PathBuf,
    name: String,
    ext: String,
    bytes: u64,
    created: Option<SystemTime>,
}

/* ───────────────────────── app state ─────────────────────────────── */

struct ViewerApp {
    images: Vec<ImgEntry>,
    files: Vec<FileMeta>,
    index_of: HashMap<PathBuf, usize>,

    current: usize,

    img_rx: Receiver<ImgMsg>,
    img_tx: Sender<ImgMsg>,

    paths_rx: Receiver<PathMsg>,
    paths_tx: Sender<PathMsg>,

    job_tx: Sender<JobMsg>,
    job_rx: Receiver<JobMsg>,

    current_dir: Option<PathBuf>,

    // We manage *borderless* fullscreen ourselves (not OS fullscreen):
    is_borderless_fs: bool,
    prev_win_pos: Option<egui::Pos2>,
    prev_win_size: Option<egui::Vec2>,

    sort_key: SortKey,
    ascending: bool,
    layout: Layout,
    show_top_bar: bool,
    prev_alt_down: bool,
    zoom: f32,
    pan: Vec2,
    last_cursor: Option<egui::Pos2>,
    pending_target: Option<String>,
    seen: HashSet<PathBuf>,
    enqueued: HashSet<PathBuf>,

    prefetch_center_idx: usize,
    prefetch_step: isize,
    prefetch_dirty: bool,

    egui_ctx: egui::Context,

    // For ultra-fast double-click detection
    last_primary_down: Option<Instant>,
}

impl ViewerApp {
    pub fn new(egui_ctx: egui::Context) -> Self {
        let (img_tx, img_rx)    = bounded::<ImgMsg>(1024);
        let (paths_tx, paths_rx)= bounded::<PathMsg>(8192);
        let (job_tx, job_rx)    = bounded::<JobMsg>(MAX_ENQUEUED_JOBS);

        Self {
            images: Vec::new(),
            files: Vec::new(),
            index_of: HashMap::new(),
            current: 0,
            img_rx, img_tx,
            paths_rx, paths_tx,
            job_tx, job_rx,
            current_dir: None,

            is_borderless_fs: false,
            prev_win_pos: None,
            prev_win_size: None,

            sort_key: SortKey::Name,
            ascending: true,
            layout: Layout::One,
            show_top_bar: true,
            prev_alt_down: false,
            zoom: 1.0,
            pan: Vec2::ZERO,
            last_cursor: None,
            pending_target: None,
            seen: HashSet::new(),
            enqueued: HashSet::new(),
            prefetch_center_idx: 0,
            prefetch_step: 0,
            prefetch_dirty: true,
            egui_ctx,
            last_primary_down: None,
        }
    }

    fn toggle_borderless_fullscreen(&mut self, ctx: &egui::Context) {
        // Use info from current viewport to get sizes/pos in *points*
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

        if !self.is_borderless_fs {
            // Save current window geometry
            self.prev_win_pos  = Some(cur_pos);
            self.prev_win_size = Some(cur_size);

            // Go borderless and cover the monitor (no OS fullscreen → no classic frame flash).
            ctx.send_viewport_cmd(egui::ViewportCommand::Decorations(false));
            ctx.send_viewport_cmd(egui::ViewportCommand::OuterPosition(egui::pos2(0.0, 0.0)));
            ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(monitor_size));
            self.is_borderless_fs = true;
        } else {
            // Restore previous geometry and decorations
            ctx.send_viewport_cmd(egui::ViewportCommand::Decorations(true));
            if let Some(s) = self.prev_win_size { ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(s)); }
            if let Some(p) = self.prev_win_pos  { ctx.send_viewport_cmd(egui::ViewportCommand::OuterPosition(p)); }
            self.is_borderless_fs = false;
        }
        ctx.request_repaint();
    }
}

/* ─────────────────── eframe integration ───────────────────────── */
impl eframe::App for ViewerApp {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        use egui::{Color32, ColorImage, CursorIcon, Key, Label, PointerButton, Pos2, Rect, Sense, Vec2};
        use std::fs;

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

        // Toggle chrome with Alt
        if input.modifiers.alt && !self.prev_alt_down { self.show_top_bar = !self.show_top_bar; }
        self.prev_alt_down = input.modifiers.alt;

        // 1) Drain decoded images → upload textures (full resolution)
        let mut uploaded = 0usize;
        loop {
            match self.img_rx.try_recv() {
                Ok((path, w, h, rgba)) => {
                    if self.seen.contains(&path) { continue; }
                    self.seen.insert(path.clone());

                    let tex = ctx.load_texture(
                        path.file_name().unwrap().to_string_lossy(),
                        ColorImage::from_rgba_unmultiplied([w, h], &rgba),
                        egui::TextureOptions::default(),
                    );
                    let (bytes, created) = fs::metadata(&path).map(|m| (m.len(), m.created().ok())).unwrap_or((0, None));

                    self.images.push(ImgEntry {
                        path: path.clone(),
                        name: tex.name().into(),
                        ext : tex.name().rsplit('.').next().unwrap_or("").into(),
                        tex, w: w as u32, h: h as u32, bytes, created,
                    });

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
            self.sort_files_lightweight();
            self.recenter_prefetch();
            ctx.request_repaint();
        }

        // 3) Prefetch scheduling
        self.schedule_prefetch_tick(PREFETCH_PER_TICK);

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
                        Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0,1.0)),
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

        // 6) Menu + stats
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
                        .show_ui(ui, |ui| {
                            for lay in Layout::all().into_iter() {
                                ui.selectable_value(&mut self.layout, lay, lay.label());
                            }
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
                        self.sort_images();
                        self.sort_files_lightweight();
                        self.zoom = 1.0; self.pan = Vec2::ZERO;
                        self.last_cursor = None;

                        self.enqueued.clear();
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

/* ───────────────────────── app methods ──────────────────────────── */

impl ViewerApp {
    fn current_img(&self) -> Option<&ImgEntry> { self.images.get(self.current) }

    fn delete_current(&mut self) {
        if self.images.is_empty() { return; }
        let path = self.images[self.current].path.clone();
        let _ = std::fs::remove_file(&path);
        self.images.remove(self.current);
        self.seen.remove(&path);
        if self.current >= self.images.len() && !self.images.is_empty() {
            self.current = self.images.len() - 1;
        }
        self.zoom = 1.; self.pan=Vec2::ZERO; self.last_cursor=None;
        self.prefetch_dirty = true;
    }

    fn sort_images(&mut self) {
        let (asc, key) = (self.ascending, self.sort_key);
        self.images.sort_by(|a,b| {
            let o = match key {
                SortKey::Name   => a.name.cmp(&b.name),
                SortKey::Created=> a.created.cmp(&b.created),
                SortKey::Size   => a.bytes.cmp(&b.bytes),
                SortKey::Height => a.h.cmp(&b.h).then(a.name.cmp(&b.name)),
                SortKey::Width  => a.w.cmp(&b.w).then(a.name.cmp(&b.name)),
                SortKey::Type   => a.ext.cmp(&b.ext).then(a.name.cmp(&b.name)),
            };
            if asc { o } else { o.reverse() }
        });
        self.current = self.current.min(self.images.len().saturating_sub(1));
    }

    fn sort_files_lightweight(&mut self) {
        let (asc, key) = (self.ascending, self.sort_key);
        self.files.sort_by(|a,b| {
            use SortKey::*;
            let o = match key {
                Name    => a.name.cmp(&b.name),
                Created => a.created.cmp(&b.created),
                Size    => a.bytes.cmp(&b.bytes),
                Type    => a.ext.cmp(&b.ext).then(a.name.cmp(&b.name)),
                Height|Width => a.name.cmp(&b.name), // fallback until decoded
            };
            if asc { o } else { o.reverse() }
        });
        self.index_of.clear();
        for (i, m) in self.files.iter().enumerate() {
            self.index_of.insert(m.path.clone(), i);
        }
    }

    fn reset_for_new_load(&mut self) {
        self.images.clear();
        self.files.clear();
        self.index_of.clear();
               self.current = 0;
        self.seen.clear();
        self.enqueued.clear();
        self.pending_target = None;
        self.prefetch_center_idx = 0;
        self.prefetch_step = 0;
        self.prefetch_dirty = true;
    }

    fn spawn_loader(&mut self, dir: PathBuf) {
        self.reset_for_new_load();
        self.current_dir = Some(dir.clone());
        self.start_decoder_workers();
        let paths_tx = self.paths_tx.clone();
        let ctx      = self.egui_ctx.clone();
        std::thread::spawn(move || enumerate_paths(dir, paths_tx, ctx));
    }

    fn spawn_loader_file(&mut self, file: PathBuf) {
        self.reset_for_new_load();
        self.start_decoder_workers();
        self.pending_target = file.file_name().and_then(|s| s.to_str()).map(|s| s.to_owned());

        if let Some(parent) = file.parent().map(|p| p.to_path_buf()) {
            self.current_dir = Some(parent.clone());

            if !self.seen.contains(&file) && self.enqueued.insert(file.clone()) {
                let _ = self.job_tx.try_send(file.clone());
            }

            let paths_tx = self.paths_tx.clone();
            let ctx      = self.egui_ctx.clone();
            std::thread::spawn(move || enumerate_paths(parent, paths_tx, ctx));
        } else {
            self.current_dir = None;
            if !self.seen.contains(&file) && self.enqueued.insert(file.clone()) {
                let _ = self.job_tx.try_send(file.clone());
            }
        }
        self.prefetch_dirty = true;
    }

    fn add_folder(&mut self, dir: PathBuf) {
        self.current_dir = Some(dir.clone());
        let paths_tx = self.paths_tx.clone();
        let ctx      = self.egui_ctx.clone();
        std::thread::spawn(move || enumerate_paths(dir, paths_tx, ctx));
        self.prefetch_dirty = true;
    }

    fn next(&mut self) {
        if self.images.is_empty() { return; }
        self.current = (self.current+1)%self.images.len();
        self.zoom=1.;self.pan=Vec2::ZERO;self.last_cursor=None;
        self.prefetch_dirty = true;
    }
    fn prev(&mut self) {
        if self.images.is_empty() { return; }
        self.current = (self.current+self.images.len()-1)%self.images.len();
        self.zoom=1.;self.pan=Vec2::ZERO;self.last_cursor=None;
        self.prefetch_dirty = true;
    }

    fn recenter_prefetch(&mut self) {
        let center_idx = if let Some(cur) = self.current_img() {
            self.index_of.get(&cur.path).copied()
        } else if let Some(t) = &self.pending_target {
            self.files.iter().position(|m| &m.name == t)
        } else { Some(0) }.unwrap_or(0);

        self.prefetch_center_idx = center_idx;
        self.prefetch_step = 0;
        self.prefetch_dirty = true;
    }

    fn schedule_prefetch_tick(&mut self, max_to_enqueue: usize) {
        if !self.prefetch_dirty || self.files.is_empty() { return; }
        if self.job_tx.len() >= MAX_ENQUEUED_JOBS { return; }

        let mut emitted = 0usize;
        while emitted < max_to_enqueue {
            let step_abs = self.prefetch_step.unsigned_abs();
            if step_abs > PREFETCH_RADIUS { break; }

            let idx = if self.prefetch_step == 0 {
                self.prefetch_center_idx as isize
            } else {
                self.prefetch_center_idx as isize + self.prefetch_step
            };

            self.prefetch_step = if self.prefetch_step <= 0 { -self.prefetch_step + 1 } else { -self.prefetch_step };

            if idx < 0 || (idx as usize) >= self.files.len() { continue; }
            let meta = &self.files[idx as usize];
            let p = &meta.path;

            if self.seen.contains(p) || self.enqueued.contains(p) { continue; }
            if self.job_tx.len() >= MAX_ENQUEUED_JOBS { break; }
            if self.enqueued.insert(p.clone()) {
                let _ = self.job_tx.try_send(p.clone());
                emitted += 1;
            }
        }

        if emitted == 0 || self.prefetch_step.unsigned_abs() > PREFETCH_RADIUS {
            self.prefetch_dirty = false;
        }
    }

    fn start_decoder_workers(&self) {
        let workers = decode_workers();
        for _ in 0..workers {
            let rx_jobs = self.job_rx.clone();
            let tx_img  = self.img_tx.clone();
            let ctx     = self.egui_ctx.clone();
            std::thread::spawn(move || {
                for path in rx_jobs.iter() {
                    if let Ok((w, h, rgba)) = decode_full_rgba(&path) {
                        let _ = tx_img.send((path, w, h, rgba));
                        ctx.request_repaint();
                    }
                }
            });
        }
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

/* ───────────────────────── background IO/decoding ──────────────── */

fn decode_workers() -> usize {
    const MIN: usize = 2;
    const MAX: usize = 8;
    let logical = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let half = (logical.max(2)) / 2;
    half.clamp(MIN, MAX)
}

fn decode_full_rgba(path: &PathBuf) -> Result<(usize, usize, Vec<u8>), ()> {
    // Decode at the image's original resolution (no preview downscaling).
    let reader = ImageReader::open(path).map_err(|_| ())?;
    let img    = reader.decode().map_err(|_| ())?;
    let di: DynamicImage = img; // keep original size
    let rgba = di.to_rgba8();
    Ok((rgba.width() as usize, rgba.height() as usize, rgba.into_raw()))
}

fn enumerate_paths(dir: PathBuf, paths_tx: Sender<PathMsg>, egui_ctx: egui::Context) {
    use std::ffi::OsStr;
    let mut sent = 0usize;

    #[inline]
    fn is_img_ext(path: &PathBuf) -> bool {
        match path.extension().and_then(OsStr::to_str).map(|s| s.to_ascii_lowercase()) {
            Some(ref e) if matches!(e.as_str(), "png" | "jpg" | "jpeg" | "bmp" | "gif" | "tiff" | "webp") => true,
            _ => false,
        }
    }

    for entry in WalkDir::new(dir).follow_links(false).min_depth(1) {
        if let Ok(e) = entry {
            if e.file_type().is_file() {
                let p = e.path().to_path_buf();
                if is_img_ext(&p) {
                    let _ = paths_tx.send(p);
                    sent += 1;
                    if sent % 64 == 0 { egui_ctx.request_repaint(); }
                }
            }
        }
    }
    egui_ctx.request_repaint();
}
