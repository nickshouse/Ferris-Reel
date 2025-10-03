use std::{
    collections::{HashMap, VecDeque},
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
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
const PATHS_PER_FRAME: usize = 64;
const IMG_BATCH_CAPACITY: usize = UPLOADS_PER_FRAME * 4;

/* Reel animation tuneables */
const REEL_GAP_PX: f32 = 12.0; // spacing between tiles
const REEL_NEIGHBORS: isize = 3; // tiles to draw on each side
const REEL_OMEGA: f32 = 12.0; // higher = snappier easing (images/sec response cap)
const REEL_SNAP_EPS: f32 = 0.2; // proximity threshold before advancing current index

/* ───────────────────────── domain types ─────────────────────────── */

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum SortKey {
    Name,
    Created,
    Size,
    Height,
    Width,
    Type,
}
impl SortKey {
    pub fn all() -> [Self; 6] {
        [
            Self::Name,
            Self::Created,
            Self::Size,
            Self::Height,
            Self::Width,
            Self::Type,
        ]
    }
    pub fn label(self) -> &'static str {
        match self {
            Self::Name => "Name",
            Self::Created => "Date",
            Self::Size => "Size",
            Self::Height => "Height",
            Self::Width => "Width",
            Self::Type => "Type",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Layout {
    One,
    Two,
    Three,
}
impl Layout {
    fn all() -> [Self; 3] {
        [Self::One, Self::Two, Self::Three]
    }
    fn label(self) -> &'static str {
        match self {
            Self::One => "1 image",
            Self::Two => "2 images",
            Self::Three => "3 images",
        }
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
    images_dirty: bool,
    files: Vec<FileMeta>,
    /// NEW: mirror of file paths in current sort order — avoids per-frame allocs/clones
    file_paths: Vec<PathBuf>,
    index_of: HashMap<PathBuf, usize>,

    current: usize,

    img_tx: Sender<ImgMsg>,
    img_batch_rx: Receiver<VecDeque<ImgMsg>>,
    img_batch_pool_tx: Sender<VecDeque<ImgMsg>>,
    pending_img_batch: Option<VecDeque<ImgMsg>>,

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
    prev_win_maximized: Option<bool>, // remember if window was maximized

    sort_key: SortKey,
    ascending: bool,
    layout: Layout,

    // UI chrome (both top and bottom panels share this single flag)
    show_top_bar: bool,
    prev_alt_down: bool,

    // Remember previous chrome visibility when entering fullscreen
    prev_chrome_visible: Option<bool>,

    // Ignore panning for a single frame after reset/toggle
    suppress_drag_once: bool,

    // Ignore panning until all pointer buttons are released after a toggle
    suppress_drag_until_release: bool,

    zoom: f32,
    pan: Vec2,
    last_cursor: Option<egui::Pos2>,
    pending_target: Option<String>,

    egui_ctx: egui::Context,

    // For ultra-fast double-click detection
    last_primary_down: Option<Instant>,

    // ── debounce expensive sort+recenter while many paths arrive
    last_sort_at: Instant,
    pending_paths_since_sort: usize,

    // ── optional animation
    animate_reel: bool,
    reel_pos: f32,           // fractional index, 0..images.len()-1 (clamped)
    reel_target: f32,        // desired index (float)
    last_anim_tick: Instant, // per-frame dt
    last_prefetch_center: Option<usize>, // last index we centered prefetch on
    reel_snap_hold: u8,
    reel_scale: f32,
    reel_scale_target: f32,
}

impl ViewerApp {
    pub fn new(egui_ctx: egui::Context) -> Self {
        let (img_tx, img_rx) = bounded::<ImgMsg>(load::IMG_CHAN_CAP);
        let (paths_tx, paths_rx) = bounded::<PathMsg>(load::PATHS_CHAN_CAP);
        let (job_tx, job_rx) = bounded::<JobMsg>(load::MAX_ENQUEUED_JOBS);
        let (job_tx_prio, job_rx_prio) = bounded::<JobMsg>(256);
        let (img_batch_tx, img_batch_rx) = bounded::<VecDeque<ImgMsg>>(2);
        let (img_batch_pool_tx, img_batch_pool_rx) = bounded::<VecDeque<ImgMsg>>(2);
        for _ in 0..2 {
            let _ = img_batch_pool_tx.send(VecDeque::with_capacity(IMG_BATCH_CAPACITY));
        }

        {
            let upstream = img_rx.clone();
            let batch_tx = img_batch_tx.clone();
            let pool_rx = img_batch_pool_rx;
            let pool_tx = img_batch_pool_tx.clone();
            std::thread::spawn(move || {
                use crossbeam_channel::RecvTimeoutError;
                use std::time::Duration;

                let mut upstream_closed = false;
                while let Ok(mut buf) = pool_rx.recv() {
                    buf.clear();

                    loop {
                        if upstream_closed {
                            break;
                        }

                        match upstream.recv_timeout(Duration::from_millis(3)) {
                            Ok(msg) => {
                                buf.push_back(msg);
                                if buf.len() >= IMG_BATCH_CAPACITY {
                                    break;
                                }
                            }
                            Err(RecvTimeoutError::Timeout) => {
                                if !buf.is_empty() {
                                    break;
                                }
                            }
                            Err(RecvTimeoutError::Disconnected) => {
                                upstream_closed = true;
                                break;
                            }
                        }
                    }

                    if buf.is_empty() {
                        if pool_tx.send(buf).is_err() {
                            break;
                        }
                        if upstream_closed {
                            break;
                        }
                        continue;
                    }

                    if batch_tx.send(buf).is_err() {
                        break;
                    }
                }
            });
        }

        Self {
            images: Vec::new(),
            images_dirty: false,
            files: Vec::new(),
            file_paths: Vec::new(),
            index_of: HashMap::new(),
            current: 0,
            img_tx,
            img_batch_rx,
            img_batch_pool_tx,
            pending_img_batch: None,
            paths_rx,
            paths_tx,
            job_tx,
            job_rx,
            job_tx_prio,
            job_rx_prio,

            qstate: load::QueueState::default(),
            // Sliding-window prefetch with reasonable default radius
            prefetch: load::Prefetcher::new(64),
            current_gen: Arc::new(AtomicU64::new(1)),

            current_dir: None,

            is_borderless_fs: false,
            prev_win_pos: None,
            prev_win_size: None,
            prev_win_maximized: None,

            sort_key: SortKey::Created,
            ascending: false,
            layout: Layout::One,

            show_top_bar: true,
            prev_alt_down: false,
            prev_chrome_visible: None,

            suppress_drag_once: false,
            suppress_drag_until_release: false,
            zoom: 1.0,
            pan: Vec2::ZERO,
            last_cursor: None,
            pending_target: None,

            egui_ctx,
            last_primary_down: None,

            // debounce state
            last_sort_at: Instant::now(),
            pending_paths_since_sort: 0,

            // animation state
            animate_reel: true,
            reel_pos: 0.0,
            reel_target: 0.0,
            last_anim_tick: Instant::now(),
            last_prefetch_center: None,
            reel_snap_hold: 0,
            reel_scale: 1.0,
            reel_scale_target: 1.0,
        }
    }

    fn launch_workers(&self, use_downscaled: bool, max_dim: u32) {
        // Dedicated decoder pool; keep walkers separate (in load::enumerate_paths).
        // Sync file IO + multi-core CPU decode (Rayon) == best bang-per-buck.
        let threads = {
            const MIN: usize = 2;
            const MAX: usize = 12;
            let logical = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4);
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
        let ctx = self.egui_ctx.clone();
        std::thread::spawn(move || load::enumerate_paths(dir, paths_tx, ctx));
    }

    pub fn spawn_loader_file(&mut self, file: PathBuf) {
        self.reset_for_new_load();
        self.launch_workers(true, 2560);
        self.pending_target = file
            .file_name()
            .and_then(|s| s.to_str())
            .map(|s| s.to_owned());

        let gen = self.current_gen.load(Ordering::Relaxed);

        if let Some(parent) = file.parent().map(|p| p.to_path_buf()) {
            self.current_dir = Some(parent.clone());

            // immediate decode for the target file (priority queue)
            let _ = self.job_tx_prio.try_send((gen, file.clone()));

            let paths_tx = self.paths_tx.clone();
            let ctx = self.egui_ctx.clone();
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
        let ctx = self.egui_ctx.clone();
        std::thread::spawn(move || load::enumerate_paths(dir, paths_tx, ctx));
        self.prefetch.dirty = true;
    }

    #[inline]
    fn reset_view_like_new(&mut self) {
        self.zoom = 1.0;
        self.pan = Vec2::ZERO;
        self.last_cursor = None;
        self.suppress_drag_once = true; // avoid drag delta in same frame
    }

    fn toggle_borderless_fullscreen(&mut self, ctx: &egui::Context) {
        let (cur_pos, cur_size, monitor_size, was_maximized) = ctx.input(|i| {
            let vp = &i.viewport();
            let cur_pos = vp
                .outer_rect
                .unwrap_or(egui::Rect::from_min_size(
                    egui::pos2(0.0, 0.0),
                    i.screen_rect().size(),
                ))
                .min;
            let cur_size = vp.inner_rect.unwrap_or(i.screen_rect()).size();
            let mon_size = vp.monitor_size.unwrap_or(i.screen_rect().size());
            // Best-effort read of maximize state (defaults to false if unknown)
            let was_maximized = vp.maximized.unwrap_or(false);
            (cur_pos, cur_size, mon_size, was_maximized)
        });

        // Reset/center the image exactly like when switching images
        self.reset_view_like_new();
        // Also stop any pan deltas until the user releases mouse/touch
        self.suppress_drag_until_release = true;

        if !self.is_borderless_fs {
            // Entering fullscreen: hide chrome, remember previous chrome/max state
            self.prev_chrome_visible = Some(self.show_top_bar);
            self.show_top_bar = false;

            self.prev_win_pos = Some(cur_pos);
            self.prev_win_size = Some(cur_size);
            self.prev_win_maximized = Some(was_maximized);

            // If currently maximized, unmaximize first so size/position commands take effect
            if was_maximized {
                ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(false));
            }

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

            let was_max = self.prev_win_maximized.take().unwrap_or(false);
            if was_max {
                // If we were maximized before entering fullscreen, just re-maximize
                ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(true));
            } else {
                // Otherwise, restore saved size/position
                if let Some(s) = self.prev_win_size {
                    ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(s));
                }
                if let Some(p) = self.prev_win_pos {
                    ctx.send_viewport_cmd(egui::ViewportCommand::OuterPosition(p));
                }
            }

            self.is_borderless_fs = false;
        }

        ctx.request_repaint();
    }

    fn current_img(&self) -> Option<&ImgEntry> {
        self.images.get(self.current)
    }

    fn delete_current(&mut self) {
        if self.images.is_empty() {
            return;
        }
        let path = self.images[self.current].path.clone();
        let _ = std::fs::remove_file(&path);
        self.images.remove(self.current);
        self.qstate.seen.remove(&path);
        if self.current >= self.images.len() && !self.images.is_empty() {
            self.current = self.images.len() - 1;
        }

        // Keep reel indices valid if animation is enabled
        if self.animate_reel {
            if !self.images.is_empty() {
                let max_idx = (self.images.len() - 1) as f32;
                self.reel_pos = self.current.min(self.images.len() - 1) as f32;
                self.reel_target = self.reel_pos.clamp(0.0, max_idx);
            } else {
                self.reel_pos = 0.0;
                self.reel_target = 0.0;
            }
        }
        self.reel_snap_hold = 0;
        self.reel_scale = 1.0;
        self.reel_scale_target = 1.0;

        self.zoom = 1.;
        self.pan = Vec2::ZERO;
        self.last_cursor = None;
        self.prefetch.dirty = true;
    }

    fn reset_for_new_load(&mut self) {
        self.images.clear();
        self.images_dirty = false;
        self.files.clear();
        self.file_paths.clear();
        self.index_of.clear();
        self.current = 0;
        self.qstate.clear();
        self.pending_target = None;
        self.prefetch.reset();
        // bump generation so stale jobs are ignored by workers
        self.current_gen.fetch_add(1, Ordering::Relaxed);
        // also clear enqueued set to allow requeue under new ordering
        self.qstate.enqueued.clear();

        // reset debounce state
        self.last_sort_at = Instant::now();
        self.pending_paths_since_sort = 0;

        // reset animation state
        self.reel_pos = 0.0;
        self.reel_target = 0.0;
        self.last_anim_tick = Instant::now();
        self.last_prefetch_center = None;
        self.reel_snap_hold = 0;
        self.reel_scale = 1.0;
        self.reel_scale_target = 1.0;
    }

    fn next(&mut self) {
        if self.images.is_empty() {
            return;
        }
        if self.animate_reel {
            let max_idx = (self.images.len() - 1) as f32;
            self.reel_target = (self.reel_target + 1.0).clamp(0.0, max_idx);
            self.last_anim_tick = Instant::now();
            self.reel_snap_hold = 2;
        } else {
            self.current = (self.current + 1) % self.images.len();
            self.reset_view_like_new();
        }
        self.prefetch.dirty = true;
    }
    fn prev(&mut self) {
        if self.images.is_empty() {
            return;
        }
        if self.animate_reel {
            let max_idx = (self.images.len() - 1) as f32;
            self.reel_target = (self.reel_target - 1.0).clamp(0.0, max_idx);
            self.last_anim_tick = Instant::now();
            self.reel_snap_hold = 2;
        } else {
            self.current = (self.current + self.images.len() - 1) % self.images.len();
            self.reset_view_like_new();
        }
        self.prefetch.dirty = true;
    }

    fn recenter_prefetch(&mut self) {
        let cur_path_owned: Option<PathBuf> = self.current_img().map(|e| e.path.clone());
        let cur_path_ref: Option<&PathBuf> = cur_path_owned.as_ref();
        let pending = self.pending_target.as_deref();
        self.prefetch
            .recenter(&self.file_paths, cur_path_ref, pending, &self.index_of);
    }

    fn fit_for_idx(&self, idx: usize, viewport: Vec2) -> f32 {
        if viewport.x <= 0.0 || viewport.y <= 0.0 || idx >= self.images.len() {
            return 1.0;
        }
        let tex_size = self.images[idx].tex.size_vec2();
        if tex_size.x <= 0.0 || tex_size.y <= 0.0 {
            return 1.0;
        }
        let fit_w = viewport.x / tex_size.x;
        let fit_h = viewport.y / tex_size.y;
        fit_w.min(fit_h).min(1.0)
    }

    fn scale_target_for(&self, center: f32, viewport: Vec2) -> f32 {
        if self.images.is_empty() {
            return 1.0;
        }
        let total = self.images.len();
        let clamped_center = center.clamp(0.0, (total - 1) as f32);
        let mut base_idx = clamped_center.floor() as usize;
        if base_idx >= total {
            base_idx = total - 1;
        }
        let frac = (clamped_center - base_idx as f32).clamp(0.0, 1.0);
        let base_fit = self.fit_for_idx(base_idx, viewport);
        if frac > 0.0 && base_idx + 1 < total {
            let next_fit = self.fit_for_idx(base_idx + 1, viewport);
            base_fit + (next_fit - base_fit) * frac
        } else {
            base_fit
        }
    }

    fn resort_images_preserve_current(&mut self) {
        if !self.images_dirty {
            return;
        }
        if self.images.len() <= 1 {
            self.images_dirty = false;
            return;
        }

        let current_path = self.current_img().map(|img| img.path.clone());
        sort::sort_images(&mut self.images, self.sort_key, self.ascending);

        if let Some(path) = current_path {
            if let Some(idx) = self.images.iter().position(|img| img.path == path) {
                self.current = idx;
            } else {
                self.current = 0;
            }
        } else {
            self.current = 0;
        }

        if self.animate_reel && !self.images.is_empty() {
            let clamped = self.current.min(self.images.len() - 1) as f32;
            self.reel_pos = clamped;
            self.reel_target = clamped;
            self.last_anim_tick = Instant::now();
            self.reel_snap_hold = 0;
        }

        self.last_prefetch_center = None;
        self.recenter_prefetch();
        self.prefetch.dirty = true;
        self.images_dirty = false;
    }

    #[inline]
    fn step_reel_animation(&mut self, viewport: Vec2) {
        if self.images.is_empty() {
            return;
        }

        if !self.animate_reel {
            let clamped = self.current.min(self.images.len() - 1) as f32;
            self.reel_pos = clamped;
            self.reel_target = clamped;
            let target = self.scale_target_for(clamped, viewport);
            self.reel_scale = target;
            self.reel_scale_target = target;
            return;
        }

        let max_idx = (self.images.len() - 1) as f32;
        self.reel_target = self.reel_target.clamp(0.0, max_idx);

        let now = Instant::now();
        let dt_total = (now - self.last_anim_tick).as_secs_f32();
        self.last_anim_tick = now;

        // 240 Hz micro-stepping for ultra-smooth motion
        let mut dt_left = dt_total.min(0.20); // safety cap in case of stalls
        const H: f32 = 1.0 / 240.0;
        while dt_left > 0.0 {
            let h = dt_left.min(H);
            let alpha = 1.0 - (-REEL_OMEGA * h).exp();
            self.reel_pos += (self.reel_target - self.reel_pos) * alpha;
            dt_left -= h;
        }

        self.reel_pos = self.reel_pos.clamp(0.0, max_idx);

        let target_scale = self.scale_target_for(self.reel_pos, viewport);
        self.reel_scale_target = target_scale;

        let mut dt_left_scale = dt_total.min(0.20);
        while dt_left_scale > 0.0 {
            let h = dt_left_scale.min(H);
            let alpha = 1.0 - (-REEL_OMEGA * h).exp();
            self.reel_scale += (self.reel_scale_target - self.reel_scale) * alpha;
            dt_left_scale -= h;
        }

        self.reel_scale = self.reel_scale.clamp(0.01, 1.0);

        let target_idx = self.reel_target.round().clamp(0.0, max_idx) as usize;
        let mut should_snap = (self.reel_target - self.reel_pos).abs() < REEL_SNAP_EPS;
        if should_snap && self.reel_snap_hold > 0 {
            self.reel_snap_hold -= 1;
            should_snap = false;
        }
        let candidate_idx = if should_snap {
            target_idx
        } else {
            self.current
        };

        if candidate_idx != self.current {
            self.current = candidate_idx;
            if should_snap {
                self.reel_snap_hold = 0;
            }
            if self.last_prefetch_center != Some(candidate_idx) {
                self.last_prefetch_center = Some(candidate_idx);
                self.prefetch.dirty = true; // keep decoders feeding the reel edges
            }
        }
    }
}

/* ─────────────────── eframe integration ───────────────────────── */
impl App for ViewerApp {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        use egui::{ColorImage, CursorIcon, Key, PointerButton};

        let input = ctx.input(|i| i.clone());

        // Ultra-fast global double-click anywhere (raw press pairing)
        {
            use std::time::Duration;
            const DC_WINDOW: Duration = Duration::from_millis(300);

            for ev in &input.events {
                if let egui::Event::PointerButton {
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    ..
                } = ev
                {
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
        if input.modifiers.alt && !self.prev_alt_down {
            self.show_top_bar = !self.show_top_bar;
        }
        self.prev_alt_down = input.modifiers.alt;

        // 1) Drain decoded images → upload textures
        let mut uploaded = 0usize;
        while uploaded < UPLOADS_PER_FRAME {
            let next_msg = match self.pending_img_batch.as_mut() {
                Some(batch) => match batch.pop_front() {
                    Some(msg) => Some(msg),
                    None => {
                        if let Some(empty) = self.pending_img_batch.take() {
                            let _ = self.img_batch_pool_tx.send(empty);
                        }
                        None
                    }
                },
                None => match self.img_batch_rx.try_recv() {
                    Ok(batch) => {
                        self.pending_img_batch = Some(batch);
                        continue;
                    }
                    Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => break,
                },
            };

            let Some((path, w, h, rgba, bytes, created)) = next_msg else {
                continue;
            };

            if self.qstate.seen.contains(&path) {
                continue;
            }

            let tex = ctx.load_texture(
                path.file_name().unwrap().to_string_lossy(),
                ColorImage::from_rgba_unmultiplied([w, h], &rgba),
                egui::TextureOptions::default(),
            );

            self.images.push(ImgEntry {
                path: path.clone(),
                name: tex.name().into(),
                ext: tex.name().rsplit('.').next().unwrap_or("").into(),
                tex,
                w: w as u32,
                h: h as u32,
                bytes,
                created,
            });
            self.images_dirty = true;

            self.qstate.mark_seen(&path);
            uploaded += 1;
        }
        if uploaded > 0 {
            if self.images_dirty {
                self.resort_images_preserve_current();
            }
            ctx.request_repaint();
        }

        // 2) Drain discovered paths → index; DEBOUNCE the expensive sort/recenter
        let mut got_paths = 0usize;
        for _ in 0..PATHS_PER_FRAME {
            match self.paths_rx.try_recv() {
                Ok(p) => {
                    let meta = lightweight_meta(&p);
                    self.index_of.insert(p.clone(), self.files.len());
                    self.files.push(meta);
                    self.file_paths.push(p); // keep mirror in-order with files[]
                    got_paths += 1;
                }
                Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => break,
            }
        }

        if got_paths > 0 {
            self.pending_paths_since_sort += got_paths;

            sort::sort_files_lightweight(
                &mut self.files,
                self.sort_key,
                self.ascending,
                &mut self.index_of,
            );

            self.file_paths.clear();
            self.file_paths
                .extend(self.files.iter().map(|m| m.path.clone()));

            self.recenter_prefetch();
            self.pending_paths_since_sort = 0;
            self.last_sort_at = Instant::now();
            ctx.request_repaint();
        }

        // 3) Prefetch scheduling — **no per-frame clones** anymore
        {
            let gen = self.current_gen.load(Ordering::Relaxed);
            let _enq = self.prefetch.tick(
                &self.file_paths,
                &mut self.qstate,
                &self.job_tx,
                load::PREFETCH_PER_TICK,
                load::MAX_ENQUEUED_JOBS,
                gen,
            );
            if _enq > 0 {
                ctx.request_repaint();
            }
        }

        // 4) Hotkeys
        if input.key_pressed(Key::ArrowRight) {
            self.next();
        }
        if input.key_pressed(Key::ArrowLeft) {
            self.prev();
        }
        if input.key_pressed(Key::F11) {
            self.toggle_borderless_fullscreen(ctx);
        }
        if input.key_pressed(Key::Escape) && self.is_borderless_fs {
            self.toggle_borderless_fullscreen(ctx);
        }
        if input.key_pressed(Key::Delete) {
            self.delete_current();
        }
        match input.raw_scroll_delta.y {
            d if d > 0.0 => self.prev(),
            d if d < 0.0 => self.next(),
            _ => {}
        }

        // 5) Central panel
        egui::CentralPanel::default().show(ctx, |ui| {
            use egui::{Color32, Pos2, Rect, Sense};
            if self.images.is_empty() {
                return;
            }

            let full = ui.available_rect_before_wrap();
            let bar_h = if self.show_top_bar { 32.0 } else { 0.0 };
            let avail = Rect::from_min_max(
                Pos2::new(full.min.x, full.min.y + bar_h),
                Pos2::new(full.max.x, full.max.y - bar_h),
            );
            let viewport = avail.size();

            if let Some(p) = input.pointer.hover_pos() {
                self.last_cursor = Some(p);
            }
            let cursor = self.last_cursor.unwrap_or(avail.center());

            // Mouse-side-button zoom
            if input.pointer.button_pressed(PointerButton::Extra2) {
                let nz = (self.zoom * 1.1).clamp(0.1, 10.0);
                self.pan += (cursor - (avail.center() + self.pan)) * (1.0 - nz / self.zoom);
                self.zoom = nz;
            }
            if input.pointer.button_pressed(PointerButton::Extra1) {
                let nz = (self.zoom / 1.1).clamp(0.1, 10.0);
                self.pan += (cursor - (avail.center() + self.pan)) * (1.0 - nz / self.zoom);
                self.zoom = nz;
            }

            let set_grab = |grabbing: bool| {
                ctx.output_mut(|o| {
                    o.cursor_icon = if grabbing {
                        CursorIcon::Grabbing
                    } else {
                        CursorIcon::Grab
                    };
                });
            };
            let pan_allowed = !(self.suppress_drag_once || self.suppress_drag_until_release);

            // Animated reel
            if self.layout == Layout::One && self.animate_reel {
                self.step_reel_animation(viewport);

                let scale = self.reel_scale.max(0.01) * self.zoom;
                let total = self.images.len();
                let center_idx_f = self.reel_pos.clamp(0.0, (total - 1) as f32);
                let mut base_idx = center_idx_f.floor() as isize;
                base_idx = base_idx.clamp(0, total as isize - 1);
                let base_idx = base_idx as usize;
                let frac = (center_idx_f - base_idx as f32).clamp(0.0, 1.0);

                let mut neighbor_indices = Vec::new();
                for k in -REEL_NEIGHBORS..=REEL_NEIGHBORS {
                    let idx_isize = base_idx as isize + k;
                    if idx_isize >= 0 && (idx_isize as usize) < total {
                        neighbor_indices.push(idx_isize as usize);
                    }
                }

                let mut tile_cache: HashMap<usize, (Vec2, egui::TextureId)> = HashMap::new();
                for idx in &neighbor_indices {
                    let entry = &self.images[*idx];
                    tile_cache.insert(*idx, (entry.tex.size_vec2() * scale, entry.tex.id()));
                }

                let step_between = |a: usize, b: usize| -> f32 {
                    let &(size_a, _) = tile_cache.get(&a).expect("tile cache missing index");
                    let &(size_b, _) = tile_cache.get(&b).expect("tile cache missing index");
                    0.5 * (size_a.x + size_b.x) + REEL_GAP_PX
                };

                let mut base_center_x = avail.center().x + self.pan.x;
                if frac > 0.0 && base_idx + 1 < total {
                    base_center_x -= frac * step_between(base_idx, base_idx + 1);
                }

                for k in (-REEL_NEIGHBORS..=REEL_NEIGHBORS).rev() {
                    let idx_isize = base_idx as isize + k;
                    if idx_isize < 0 || (idx_isize as usize) >= total {
                        continue;
                    }
                    let idx = idx_isize as usize;

                    let &(size, tex_id) = match tile_cache.get(&idx) {
                        Some(entry) => entry,
                        None => continue,
                    };

                    let mut cx = base_center_x;
                    if idx < base_idx {
                        let mut j = idx;
                        while j < base_idx {
                            cx -= step_between(j, j + 1);
                            j += 1;
                        }
                    } else if idx > base_idx {
                        let mut j = base_idx;
                        while j < idx {
                            cx += step_between(j, j + 1);
                            j += 1;
                        }
                    }

                    let cy = avail.center().y + self.pan.y;
                    let rect = Rect::from_center_size(Pos2::new(cx, cy), size);

                    let resp = ui.allocate_rect(rect, Sense::click_and_drag());
                    ui.painter().image(
                        tex_id,
                        rect,
                        egui::Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                        Color32::WHITE,
                    );

                    if resp.hovered() {
                        set_grab(false);
                    }
                    if resp.dragged() {
                        if pan_allowed {
                            self.pan += resp.drag_delta();
                            set_grab(true);
                        }
                    }
                }
            } else {
                // Legacy rendering
                if self.layout == Layout::One {
                    self.step_reel_animation(viewport);
                }
                match self.layout {
                    Layout::One => {
                        let img = self.current_img().unwrap();
                        let base = img.tex.size_vec2();
                        let fit = (avail.width() / base.x)
                            .min(avail.height() / base.y)
                            .min(1.0);
                        let size = base * fit * self.zoom;
                        let rect = Rect::from_center_size(avail.center() + self.pan, size);

                        let resp = ui.allocate_rect(rect, Sense::click_and_drag());

                        ui.painter().image(
                            img.tex.id(),
                            rect,
                            egui::Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                            Color32::WHITE,
                        );

                        if resp.hovered() {
                            set_grab(false);
                        }
                        if resp.dragged() {
                            if pan_allowed {
                                self.pan += resp.drag_delta();
                                set_grab(true);
                            }
                        }
                    }
                    Layout::Two | Layout::Three => {
                        let n = if self.layout == Layout::Two { 2 } else { 3 };
                        let mut tiles = Vec::with_capacity(n);
                        for i in 0..n {
                            let e = &self.images[(self.current + i) % self.images.len()];
                            let fit = (avail.width() / (n as f32) / e.tex.size_vec2().x)
                                .min(avail.height() / e.tex.size_vec2().y)
                                .min(1.0);
                            tiles.push((e.tex.id(), e.tex.size_vec2() * fit * self.zoom));
                        }
                        let w: f32 = tiles.iter().map(|(_, s)| s.x).sum();
                        let h: f32 = tiles.iter().map(|(_, s)| s.y).fold(0.0, f32::max);
                        let rect =
                            Rect::from_center_size(avail.center() + self.pan, Vec2::new(w, h));

                        let resp = ui.allocate_rect(rect, Sense::click_and_drag());

                        ui.allocate_ui_at_rect(rect, |ui| {
                            ui.horizontal_centered(|ui| {
                                for (id, sz) in tiles {
                                    ui.image((id, sz));
                                }
                            });
                        });

                        if resp.hovered() {
                            set_grab(false);
                        }
                        if resp.dragged() {
                            if pan_allowed {
                                self.pan += resp.drag_delta();
                                set_grab(true);
                            }
                        }
                    }
                }
            }
        });

        // 6) Menu + stats (hidden when show_top_bar is false)
        if self.show_top_bar {
            egui::TopBottomPanel::top("menu").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Open file…").clicked() {
                        if let Some(f) = rfd::FileDialog::new()
                            .add_filter(
                                "Images",
                                &["png", "jpg", "jpeg", "bmp", "gif", "tiff", "webp"],
                            )
                            .pick_file()
                        {
                            self.spawn_loader_file(f);
                        }
                    }
                    if ui.button("Add folder…").clicked() {
                        if let Some(d) = rfd::FileDialog::new().pick_folder() {
                            self.add_folder(d);
                        }
                    }
                    ui.separator();

                    // Animation toggle
                    let prev_anim = self.animate_reel;
                    ui.checkbox(&mut self.animate_reel, "Animation");
                    if self.animate_reel != prev_anim {
                        if self.animate_reel {
                            self.reel_pos = self.current as f32;
                            self.reel_target = self.reel_pos;
                            self.last_anim_tick = Instant::now();
                            self.last_prefetch_center = None;
                            self.prefetch.dirty = true;
                            self.reel_snap_hold = 0;
                        } else {
                            self.current = self.reel_pos.round() as usize;
                            self.prefetch.dirty = true;
                            self.reel_snap_hold = 0;
                        }
                    }

                    ui.separator();

                    let prev = (self.layout, self.sort_key, self.ascending);
                    ui.label("View:");
                    egui::ComboBox::from_id_source("layout")
                        .selected_text(self.layout.label())
                        .show_ui(ui, |ui| {
                            for lay in Layout::all() {
                                ui.selectable_value(&mut self.layout, lay, lay.label());
                            }
                        });
                    ui.separator();
                    ui.label("Sort by:");
                    egui::ComboBox::from_id_source("sort_key")
                        .selected_text(self.sort_key.label())
                        .show_ui(ui, |ui| {
                            for k in SortKey::all() {
                                ui.selectable_value(&mut self.sort_key, k, k.label());
                            }
                        });
                    ui.label("Order:");
                    egui::ComboBox::from_id_source("sort_ord")
                        .selected_text(if self.ascending { "Asc" } else { "Desc" })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.ascending, true, "Asc");
                            ui.selectable_value(&mut self.ascending, false, "Desc");
                        });

                    if prev != (self.layout, self.sort_key, self.ascending) {
                        sort::sort_images(&mut self.images, self.sort_key, self.ascending);
                        sort::sort_files_lightweight(
                            &mut self.files,
                            self.sort_key,
                            self.ascending,
                            &mut self.index_of,
                        );

                        // keep file_paths in sync with new order
                        self.file_paths.clear();
                        self.file_paths
                            .extend(self.files.iter().map(|m| m.path.clone()));

                        if self.animate_reel {
                            self.reel_pos = self.current as f32;
                            self.reel_target = self.reel_pos;
                            self.reel_snap_hold = 0;
                        }
                        self.zoom = 1.0;
                        self.pan = Vec2::ZERO;
                        self.last_cursor = None;

                        self.qstate.enqueued.clear();
                        self.recenter_prefetch();
                        self.prefetch.dirty = true;
                    }

                    ui.separator();
                    if ui
                        .add_enabled(!self.images.is_empty(), egui::Button::new("< Prev"))
                        .clicked()
                    {
                        self.prev();
                    }
                    if ui
                        .add_enabled(!self.images.is_empty(), egui::Button::new("Next >"))
                        .clicked()
                    {
                        self.next();
                    }
                });
            });

            egui::TopBottomPanel::bottom("stats").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if let Some(dir) = &self.current_dir {
                        ui.label(format!("Last folder: {}", dir.display()));
                        ui.separator();
                    }
                    if let Some(img) = self.current_img() {
                        ui.label(&img.name);
                        ui.separator();
                        ui.label(format!("{} / {}", self.current + 1, self.images.len()));
                        ui.separator();
                        ui.label(format!("{}×{}", img.w, img.h));
                        ui.separator();
                        ui.label(human_bytes(img.bytes));
                    }
                });
            });
        }

        if self.suppress_drag_once {
            self.suppress_drag_once = false;
        }
        if self.suppress_drag_until_release && !input.pointer.any_down() {
            self.suppress_drag_until_release = false;
        }

        // 7) Drive animations even when idle — bump to ~120 Hz for fluidity
        {
            use std::time::Duration;
            const EPS: f32 = 0.0005;
            let anim_running = self.animate_reel
                && !self.images.is_empty()
                && (self.reel_target - self.reel_pos).abs() > EPS;

            if anim_running {
                ctx.request_repaint_after(Duration::from_secs_f32(1.0 / 120.0));
            }
        }

        // Default cursor if we didn't set Grab/Grabbing above:
        ctx.output_mut(|o| {
            if !matches!(o.cursor_icon, CursorIcon::Grab | CursorIcon::Grabbing) {
                o.cursor_icon = CursorIcon::Default;
            }
        });
    }
}

/* ───────────────────────── helpers ──────────────────────────── */

fn lightweight_meta(path: &PathBuf) -> FileMeta {
    use std::fs;
    let (bytes, created) = fs::metadata(path)
        .map(|m| (m.len(), m.created().ok()))
        .unwrap_or((0, None));
    let name = path
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let ext = path
        .extension()
        .map(|s| s.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    FileMeta {
        path: path.clone(),
        name,
        ext,
        bytes,
        created,
    }
}

fn human_bytes(b: u64) -> String {
    let f = b as f64;
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    if f >= GB {
        format!("{:.1} GB", f / GB)
    } else if f >= MB {
        format!("{:.1} MB", f / MB)
    } else if f >= KB {
        format!("{:.1} KB", f / KB)
    } else {
        format!("{b} B")
    }
}
