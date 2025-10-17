use std::{
    collections::{HashMap, VecDeque},
    io,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Instant, SystemTime},
};

#[cfg(target_os = "windows")]
use std::process::Command;

use crossbeam_channel::{bounded, Receiver, Sender, TryRecvError};
use eframe::{egui, egui::TextureHandle, App};
use egui::Vec2;

use crate::load::{self, ImgMsg, JobMsg, PathMsg};
use crate::sort;

/* ───────────────────────── UI tuneables ─────────────────────────── */

const UPLOADS_PER_FRAME: usize = 4;
const PATHS_PER_FRAME: usize = 64;
const IMG_BATCH_CAPACITY: usize = UPLOADS_PER_FRAME * 4;

/* Crossfade for single-image mode */
const XFADE_SECS: f32 = 0.16; // set 0.0 for hard snap

/* Reel tuneables (all GUI-thread driven) */
const REEL_GAP_PX: f32 = 12.0; // gap between tiles
const REEL_NEIGHBORS: isize = 3; // tiles to draw on each side when showing neighbors
const REEL_OMEGA: f32 = 14.0; // responsiveness (larger = snappier)
const REEL_SNAP_EPS: f32 = 0.15; // when |target - pos| < eps, snap current
const REEL_SCROLL_SENS: f32 = 0.01; // how many indices per scroll-point
const REEL_SCROLL_CLAMP: f32 = 3.0; // cap per-frame scroll adjustment

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

/* ───────────────────────── transitions ─────────────────────────── */

#[derive(Clone)]
struct Transition {
    from_idx: usize,
    to_idx: usize,
    start: Instant,
    dur: f32, // seconds
}
impl Transition {
    fn progress(&self) -> f32 {
        let t = (Instant::now() - self.start).as_secs_f32();
        (t / self.dur.max(1e-6)).clamp(0.0, 1.0)
    }
    fn smoothstep(p: f32) -> f32 {
        p * p * (3.0 - 2.0 * p)
    }
}

enum ImageMenuAction {
    OpenFolder(PathBuf),
    EditNotepad(PathBuf),
}

/* ───────────────────────── app state ─────────────────────────────── */

pub struct ViewerApp {
    images: Vec<ImgEntry>,
    images_dirty: bool,
    files: Vec<FileMeta>,
    /// Mirror of file paths in current sort order — avoids per-frame allocs/clones
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
    top_bar_rect: Option<egui::Rect>,
    bottom_bar_rect: Option<egui::Rect>,

    // Ignore panning for a single frame after reset/toggle
    suppress_drag_once: bool,

    // Ignore panning until all pointer buttons are released after a toggle
    suppress_drag_until_release: bool,

    // Deferred reset to cover viewport changes (processed at frame start)
    pending_reset_pan_zoom: bool,

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

    // Single-image transition
    transition: Option<Transition>,

    // Reel mode (carousel)
    reel_enabled: bool,
    reel_pos: f32,           // fractional index
    reel_target: f32,        // desired index
    last_anim_tick: Instant, // frame delta clock
    last_prefetch_center: Option<usize>,
    reel_snap_hold: u8,     // small debounce frames after keyboard jumps
    reel_scroll_accum: f32, // accumulated scroll fractions
    reel_auto_speed: f32,   // images-per-second auto scroll rate
    reel_auto_phase: f32,   // running phase for auto scroll wrapping
    reel_looping: bool,
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

        // Image message coalescer thread
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
            top_bar_rect: None,
            bottom_bar_rect: None,

            suppress_drag_once: false,
            suppress_drag_until_release: false,
            pending_reset_pan_zoom: false,
            zoom: 1.0,
            pan: Vec2::ZERO,
            last_cursor: None,
            pending_target: None,

            egui_ctx: egui_ctx,
            last_primary_down: None,

            last_sort_at: Instant::now(),
            pending_paths_since_sort: 0,

            transition: None,

            reel_enabled: false, // default OFF so reel is opt-in
            reel_pos: 0.0,
            reel_target: 0.0,
            last_anim_tick: Instant::now(),
            last_prefetch_center: None,
            reel_snap_hold: 0,
            reel_scroll_accum: 0.0,
            reel_auto_speed: 0.0,
            reel_auto_phase: 0.0,
            reel_looping: false,
        }
    }

    fn launch_workers(&self, use_downscaled: bool, max_dim: u32) {
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

    fn resort_images_preserve_current(&mut self) {
        if !self.images_dirty {
            return;
        }
        if self.images.len() <= 1 {
            self.images_dirty = false;
            return;
        }

        // Remember current by path, so we can find it again after sort
        let current_path = self.current_img().map(|img| img.path.clone());

        // Sort the loaded images by the current key/order
        sort::sort_images(&mut self.images, self.sort_key, self.ascending);

        // Restore current index to the same image if it still exists
        if let Some(path) = current_path {
            if let Some(idx) = self.images.iter().position(|img| img.path == path) {
                self.current = idx;
            } else {
                self.current = 0;
            }
        } else {
            self.current = 0;
        }

        // Sorting invalidates any from/to indices for crossfade
        self.transition = None;

        // Keep reel indices coherent
        if self.reel_enabled && !self.images.is_empty() {
            let clamped = self.current.min(self.images.len() - 1) as f32;
            self.reel_pos = clamped;
            self.reel_target = clamped;
            self.reel_snap_hold = 0;
        }

        // Recenter prefetch around the (possibly moved) current
        self.last_prefetch_center = None;
        self.recenter_prefetch();
        self.prefetch.dirty = true;

        self.images_dirty = false;
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
    fn reset_reel_offsets(&mut self) {
        if !self.reel_enabled {
            return;
        }
        if self.images.is_empty() {
            self.reel_target = 0.0;
            self.reel_pos = 0.0;
        } else {
            let anchor = self.current.min(self.reel_max_start());
            self.reel_target = anchor as f32;
            self.reel_pos = self.reel_target;
        }
        self.reel_scroll_accum = 0.0;
        self.reel_snap_hold = 0;
    }

    #[inline]
    fn reset_pan_zoom(&mut self) {
        self.zoom = 1.0;
        self.pan = Vec2::ZERO;
        self.last_cursor = None;
        self.reset_reel_offsets();
        self.pending_reset_pan_zoom = false;
    }

    #[inline]
    fn reset_view_like_new(&mut self) {
        self.reset_pan_zoom();
        self.suppress_drag_once = true;
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
            let was_maximized = vp.maximized.unwrap_or(false);
            (cur_pos, cur_size, mon_size, was_maximized)
        });

        self.reset_view_like_new();
        self.suppress_drag_until_release = true;

        if !self.is_borderless_fs {
            self.prev_chrome_visible = Some(self.show_top_bar);
            self.show_top_bar = false;

            self.prev_win_pos = Some(cur_pos);
            self.prev_win_size = Some(cur_size);
            self.prev_win_maximized = Some(was_maximized);

            if was_maximized {
                ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(false));
            }

            ctx.send_viewport_cmd(egui::ViewportCommand::Decorations(false));
            ctx.send_viewport_cmd(egui::ViewportCommand::OuterPosition(egui::pos2(0.0, 0.0)));
            ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(monitor_size));
            self.is_borderless_fs = true;
        } else {
            if let Some(prev) = self.prev_chrome_visible.take() {
                self.show_top_bar = prev;
            } else {
                self.show_top_bar = true;
            }

            ctx.send_viewport_cmd(egui::ViewportCommand::Decorations(true));

            let was_max = self.prev_win_maximized.take().unwrap_or(false);
            if was_max {
                ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(true));
            } else {
                if let Some(s) = self.prev_win_size {
                    ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(s));
                }
                if let Some(p) = self.prev_win_pos {
                    ctx.send_viewport_cmd(egui::ViewportCommand::OuterPosition(p));
                }
            }

            self.is_borderless_fs = false;
        }

        self.reset_pan_zoom();
        self.pending_reset_pan_zoom = true;
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
        let mut removed = false;
        match trash::delete(&path) {
            Ok(()) => {
                removed = true;
            }
            Err(err) => {
                eprintln!(
                    "Failed to move {} to recycle bin: {err}",
                    path.display()
                );
                match std::fs::remove_file(&path) {
                    Ok(()) => {
                        removed = true;
                    }
                    Err(fs_err) => {
                        eprintln!(
                            "Fallback remove_file failed for {}: {fs_err}",
                            path.display()
                        );
                    }
                }
            }
        }

        if !removed {
            return;
        }
        self.images.remove(self.current);
        self.qstate.seen.remove(&path);
        if self.current >= self.images.len() && !self.images.is_empty() {
            self.current = self.images.len() - 1;
        }
        self.transition = None;

        self.reset_pan_zoom();
        self.prefetch.dirty = true;
    }

    fn image_context_menu(response: &egui::Response, path: PathBuf) -> Option<ImageMenuAction> {
        #[derive(Clone, Copy)]
        enum Choice {
            OpenFolder,
            EditNotepad,
        }
        let mut choice: Option<Choice> = None;
        response.context_menu(|ui| {
            if ui.button("Open containing folder").clicked() {
                choice = Some(Choice::OpenFolder);
                ui.close_menu();
            }
            if ui.button("Edit with Notepad").clicked() {
                choice = Some(Choice::EditNotepad);
                ui.close_menu();
            }
        });
        choice.map(|c| match c {
            Choice::OpenFolder => ImageMenuAction::OpenFolder(path),
            Choice::EditNotepad => ImageMenuAction::EditNotepad(path),
        })
    }

    fn process_image_actions(actions: Vec<ImageMenuAction>) {
        for action in actions {
            match action {
                ImageMenuAction::OpenFolder(path) => {
                    if let Err(err) = Self::open_containing_folder(&path) {
                        eprintln!(
                            "Failed to open containing folder for {}: {err}",
                            path.display()
                        );
                    }
                }
                ImageMenuAction::EditNotepad(path) => {
                    if let Err(err) = Self::edit_with_notepad(&path) {
                        eprintln!("Failed to open Notepad for {}: {err}", path.display());
                    }
                }
            }
        }
    }

    #[cfg(target_os = "windows")]
    fn open_containing_folder(path: &Path) -> io::Result<()> {
        Command::new("explorer.exe")
            .arg("/select,")
            .arg(path)
            .spawn()
            .map(|_| ())
    }

    #[cfg(not(target_os = "windows"))]
    fn open_containing_folder(path: &Path) -> io::Result<()> {
        let _ = path;
        Err(io::Error::new(
            io::ErrorKind::Other,
            "Open containing folder is only supported on Windows",
        ))
    }

    #[cfg(target_os = "windows")]
    fn edit_with_notepad(path: &Path) -> io::Result<()> {
        Command::new("notepad.exe").arg(path).spawn().map(|_| ())
    }

    #[cfg(not(target_os = "windows"))]
    fn edit_with_notepad(path: &Path) -> io::Result<()> {
        let _ = path;
        Err(io::Error::new(
            io::ErrorKind::Other,
            "Edit with Notepad is only supported on Windows",
        ))
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
        self.current_gen.fetch_add(1, Ordering::Relaxed);
        self.qstate.enqueued.clear();

        self.last_sort_at = Instant::now();
        self.pending_paths_since_sort = 0;

        self.transition = None;

        // Reel reset
        self.reel_pos = 0.0;
        self.reel_target = 0.0;
        self.last_prefetch_center = None;
        self.reel_snap_hold = 0;
        self.reel_scroll_accum = 0.0;
        self.reel_auto_phase = self.reel_target;
    }

    fn next(&mut self) {
        if self.images.len() <= 1 {
            return;
        }
        if self.reel_enabled {
            let visible = self.visible_per_page().min(self.images.len());
            if visible == 0 {
                return;
            }
            let max_start = self.reel_max_start() as f32;
            let step = visible as f32;
            if self.reel_looping {
                self.reel_target += step;
            } else {
                self.reel_target = (self.reel_target + step).clamp(0.0, max_start);
            }
            self.reel_snap_hold = 2;
            self.reel_scroll_accum = 0.0;
            self.reel_auto_phase = self.reel_target;
        } else {
            let visible = self.visible_per_page().min(self.images.len()).max(1);
            let from = self.current;
            if visible == 1 {
                let to = (self.current + 1) % self.images.len();
                self.current = to;
                if XFADE_SECS > 0.0 {
                    self.transition = Some(Transition {
                        from_idx: from,
                        to_idx: to,
                        start: Instant::now(),
                        dur: XFADE_SECS,
                    });
                } else {
                    self.transition = None;
                }
            } else {
                if visible >= self.images.len() {
                    self.current = 0;
                } else {
                    let mut next_start = self.current + visible;
                    if next_start >= self.images.len() {
                        next_start = 0;
                    }
                    self.current = next_start;
                }
                if XFADE_SECS > 0.0 && self.current != from {
                    self.transition = Some(Transition {
                        from_idx: from,
                        to_idx: self.current,
                        start: Instant::now(),
                        dur: XFADE_SECS,
                    });
                } else {
                    self.transition = None;
                }
            }
            if self.current != from {
                self.reset_view_like_new();
            }
        }
        self.prefetch.dirty = true;
    }
    fn prev(&mut self) {
        if self.images.len() <= 1 {
            return;
        }
        if self.reel_enabled {
            let visible = self.visible_per_page().min(self.images.len());
            if visible == 0 {
                return;
            }
            let max_start = self.reel_max_start() as f32;
            let step = visible as f32;
            if self.reel_looping {
                self.reel_target -= step;
            } else {
                self.reel_target = (self.reel_target - step).clamp(0.0, max_start);
            }
            self.reel_snap_hold = 2;
            self.reel_scroll_accum = 0.0;
            self.reel_auto_phase = self.reel_target;
        } else {
            let visible = self.visible_per_page().min(self.images.len()).max(1);
            let from = self.current;
            if visible == 1 {
                let to = (self.current + self.images.len() - 1) % self.images.len();
                self.current = to;
                if XFADE_SECS > 0.0 {
                    self.transition = Some(Transition {
                        from_idx: from,
                        to_idx: to,
                        start: Instant::now(),
                        dur: XFADE_SECS,
                    });
                } else {
                    self.transition = None;
                }
            } else {
                if visible >= self.images.len() {
                    self.current = 0;
                } else if self.current == 0 {
                    let remainder = self.images.len() % visible;
                    if remainder == 0 {
                        self.current = self.images.len() - visible;
                    } else {
                        self.current = self.images.len() - remainder;
                    }
                } else {
                    self.current = self.current.saturating_sub(visible);
                }
                if XFADE_SECS > 0.0 && self.current != from {
                    self.transition = Some(Transition {
                        from_idx: from,
                        to_idx: self.current,
                        start: Instant::now(),
                        dur: XFADE_SECS,
                    });
                } else {
                    self.transition = None;
                }
            }
            if self.current != from {
                self.reset_view_like_new();
            }
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

    #[inline]
    fn visible_per_page(&self) -> usize {
        match self.layout {
            Layout::One => 1,
            Layout::Two => 2,
            Layout::Three => 3,
        }
    }

    #[inline]
    fn reel_max_start(&self) -> usize {
        if self.images.is_empty() {
            0
        } else {
            let visible = self.visible_per_page().min(self.images.len());
            self.images.len().saturating_sub(visible)
        }
    }

    fn align_reel_to_layout(&mut self) {
        if !self.reel_enabled || self.images.is_empty() {
            self.reel_pos = 0.0;
            self.reel_target = 0.0;
            self.current = self.current.min(self.images.len().saturating_sub(1));
            return;
        }

        let visible = self.visible_per_page().min(self.images.len());
        if visible == 0 {
            self.reel_pos = 0.0;
            self.reel_target = 0.0;
            self.current = 0;
            return;
        }

        let max_start = self.reel_max_start();
        let anchor = self.current.min(max_start);

        self.current = anchor;
        self.reel_target = anchor as f32;
        self.reel_pos = self.reel_target;
        self.last_prefetch_center = None;
        self.reel_snap_hold = 0;
        self.reel_scroll_accum = 0.0;
        self.reel_auto_phase = self.reel_target;
    }

    #[inline]
    fn step_reel_animation(&mut self, ctx: &egui::Context, _viewport: Vec2) {
        if !self.reel_enabled || self.images.is_empty() {
            return;
        }

        let total = self.images.len();
        let visible = self.visible_per_page().min(total).max(1);
        let max_start = self.reel_max_start() as f32;

        let now = Instant::now();
        let dt = (now - self.last_anim_tick).as_secs_f32().min(0.25);
        self.last_anim_tick = now;

        let auto_speed = if self.reel_auto_speed.is_finite() {
            self.reel_auto_speed.max(0.0)
        } else {
            0.0
        };

        let auto_enabled = auto_speed > 0.0001;
        let loop_mode = self.reel_looping && auto_enabled && total > 0;

        if loop_mode {
            self.reel_target += auto_speed * dt;
        } else if auto_enabled && max_start > 0.0 {
            self.reel_target = (self.reel_target + auto_speed * dt).min(max_start);
        }

        if !loop_mode {
            self.reel_target = self.reel_target.clamp(0.0, max_start);
        }

        let alpha = 1.0 - (-REEL_OMEGA * dt).exp();
        self.reel_pos += (self.reel_target - self.reel_pos) * alpha;

        if !loop_mode {
            self.reel_pos = self.reel_pos.clamp(0.0, max_start);
        }

        if loop_mode {
            self.reel_auto_phase = self.reel_target;
            let stride = total.max(1) as f32;
            if self.reel_target.abs() > stride * 1000.0 {
                let shift = (self.reel_target / stride).trunc() * stride;
                if shift != 0.0 {
                    self.reel_target -= shift;
                    self.reel_pos -= shift;
                    self.reel_auto_phase -= shift;
                }
            }
        } else {
            self.reel_auto_phase = self.reel_target;
        }

        // snap + current index + prefetch
        let target_idx = if loop_mode {
            wrap_index(self.reel_target.round() as i64, total)
        } else {
            self.reel_target.round().clamp(0.0, max_start) as usize
        };
        let mut should_snap = (self.reel_target - self.reel_pos).abs() < REEL_SNAP_EPS;
        if should_snap && self.reel_snap_hold > 0 {
            self.reel_snap_hold -= 1;
            should_snap = false;
        }
        let candidate_idx = if should_snap {
            target_idx
        } else if loop_mode {
            wrap_index(self.reel_pos.round() as i64, total)
        } else {
            self.current.min(self.reel_max_start())
        };
        if candidate_idx != self.current {
            self.current = candidate_idx;
            if should_snap {
                self.reel_snap_hold = 0;
            }
            let mut prefetch_center = candidate_idx + visible / 2;
            if prefetch_center >= total {
                prefetch_center %= total.max(1);
            }
            if self.last_prefetch_center != Some(prefetch_center) {
                self.last_prefetch_center = Some(prefetch_center);
                self.prefetch.dirty = true;
            }
        }

        const EPS: f32 = 0.0005;
        if loop_mode || (self.reel_target - self.reel_pos).abs() > EPS {
            ctx.request_repaint(); // vsync paced
        }
    }
}

/* ─────────────────── eframe integration ───────────────────────── */
impl App for ViewerApp {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        use egui::{ColorImage, CursorIcon, Key, PointerButton};

        let input = ctx.input(|i| i.clone());

        if self.pending_reset_pan_zoom {
            self.reset_pan_zoom();
            self.suppress_drag_once = true;
            self.suppress_drag_until_release = true;
        }

        // Ultra-fast global double-click anywhere (raw press pairing)
        {
            use std::time::Duration;
            const DC_WINDOW: Duration = Duration::from_millis(300);

            for ev in &input.events {
                if let egui::Event::PointerButton {
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    pos,
                    ..
                } = ev
                {
                    let in_chrome = self.top_bar_rect.map_or(false, |r| r.contains(*pos))
                        || self.bottom_bar_rect.map_or(false, |r| r.contains(*pos));
                    if in_chrome {
                        self.last_primary_down = None;
                        continue;
                    }

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

        /* Is reel moving or crossfade active? Throttle heavy work while in motion. */
        let auto_reel_active = self.reel_looping
            && self.reel_enabled
            && self.reel_auto_speed > 0.0001
            && self.reel_max_start() > 0;
        let reel_motion = (self.reel_target - self.reel_pos).abs() > 0.0005;
        let reel_active = self.reel_enabled
            && !self.images.is_empty()
            && (reel_motion || auto_reel_active);
        let xfade_active = self
            .transition
            .as_ref()
            .map_or(false, |t| t.progress() < 1.0);
        let anim_active = reel_active || xfade_active;

        // 1) Drain decoded images → upload textures (throttle during animation)
        let uploads_cap = if anim_active { 1 } else { UPLOADS_PER_FRAME };
        let mut uploaded = 0usize;
        while uploaded < uploads_cap {
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

        // 2) Drain discovered paths → index
        let mut got_paths = 0usize;
        for _ in 0..PATHS_PER_FRAME {
            match self.paths_rx.try_recv() {
                Ok(p) => {
                    let meta = lightweight_meta(&p);
                    self.index_of.insert(p.clone(), self.files.len());
                    self.files.push(meta);
                    self.file_paths.push(p);
                    got_paths += 1;
                }
                Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => break,
            }
        }

        if got_paths > 0 {
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

        // 3) Prefetch scheduling (pause during animation to reduce contention)
        {
            let gen = self.current_gen.load(Ordering::Relaxed);
            let budget = if anim_active {
                0
            } else {
                load::PREFETCH_PER_TICK
            };
            let _enq = if budget == 0 {
                0
            } else {
                self.prefetch.tick(
                    &self.file_paths,
                    &mut self.qstate,
                    &self.job_tx,
                    budget,
                    load::MAX_ENQUEUED_JOBS,
                    gen,
                )
            };
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
        let mut handled_scroll = false;
        if self.reel_enabled && !self.images.is_empty() {
            let visible = self.visible_per_page().min(self.images.len());
            if visible > 0 {
                let smooth_scroll = if input.smooth_scroll_delta.y != 0.0 {
                    input.smooth_scroll_delta.y
                } else {
                    input.raw_scroll_delta.y
                };
                if smooth_scroll != 0.0 {
                    self.reel_scroll_accum += smooth_scroll * REEL_SCROLL_SENS;
                }

                let max_steps = REEL_SCROLL_CLAMP.max(1.0).ceil() as i32;
                let mut steps = 0;
                while self.reel_scroll_accum <= -1.0 && steps < max_steps {
                    self.reel_scroll_accum += 1.0;
                    steps += 1;
                }
                while self.reel_scroll_accum >= 1.0 && steps > -max_steps {
                    self.reel_scroll_accum -= 1.0;
                    steps -= 1;
                }

                if steps != 0 {
                    let limited = steps.clamp(-1, 1);
                    if limited != steps {
                        self.reel_scroll_accum -= (steps - limited) as f32;
                        steps = limited;
                    }
                    self.reel_scroll_accum = self.reel_scroll_accum.clamp(-0.99_f32, 0.99_f32);
                    let max_start = self.reel_max_start() as f32;
                    let prev_target = self.reel_target;
                    let delta = steps as f32 * visible as f32;
                    let new_target = if self.reel_looping {
                        self.reel_target + delta
                    } else {
                        (self.reel_target + delta).clamp(0.0, max_start)
                    };
                    handled_scroll = true;
                    if (new_target - prev_target).abs() > f32::EPSILON {
                        self.reel_target = new_target;
                        self.reel_auto_phase = self.reel_target;
                        ctx.request_repaint();
                    }
                }
            } else {
                self.reel_scroll_accum = 0.0;
            }
        } else {
            self.reel_scroll_accum = 0.0;
        }
        if !handled_scroll {
            match input.raw_scroll_delta.y {
                d if d > 0.0 => self.prev(),
                d if d < 0.0 => self.next(),
                _ => {}
            }
        }

        /* 5) PANELS FIRST: draw top/bottom now so CentralPanel gets the remaining space */
        if self.show_top_bar {
            let top_panel = egui::TopBottomPanel::top("menu").show(ctx, |ui| {
                ui.with_layer_id(
                    egui::LayerId::new(egui::Order::Foreground, egui::Id::new("menu_widgets")),
                    |ui| {
                        ui.horizontal(|ui| {
                            if ui.button("Open file...").clicked() {
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
                            if ui.button("Add folder...").clicked() {
                                if let Some(d) = rfd::FileDialog::new().pick_folder() {
                                    self.add_folder(d);
                                }
                            }
                            ui.separator();

                            // View/sort controls
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

                            // Reel toggle
                            ui.separator();
                            let reel_toggle = ui.checkbox(&mut self.reel_enabled, "Reel");
                            if reel_toggle.changed() {
                                if self.reel_enabled {
                                    self.align_reel_to_layout();
                                } else {
                                    self.reel_looping = false;
                                }
                                self.reel_auto_phase = self.reel_target;
                                self.transition = None;
                            }
                            ui.separator();
                            let auto_slider = egui::Slider::new(&mut self.reel_auto_speed, 0.0..=5.0)
                                .text("Auto scroll");
                            ui.add_enabled(self.reel_enabled, auto_slider);
                            let loop_widget = ui
                                .add_enabled(self.reel_enabled, egui::Checkbox::new(&mut self.reel_looping, "Loop"));
                            if loop_widget.changed() {
                                self.reel_auto_phase = self.reel_target;
                                self.reel_pos = self.reel_target;
                            }
                            if !self.reel_enabled {
                                self.reel_looping = false;
                            }

                            if prev != (self.layout, self.sort_key, self.ascending) {
                                sort::sort_images(&mut self.images, self.sort_key, self.ascending);
                                sort::sort_files_lightweight(
                                    &mut self.files,
                                    self.sort_key,
                                    self.ascending,
                                    &mut self.index_of,
                                );
                                self.file_paths.clear();
                                self.file_paths
                                    .extend(self.files.iter().map(|m| m.path.clone()));
                                self.reset_pan_zoom();
                                self.transition = None;
                                self.qstate.enqueued.clear();
                                if self.reel_enabled {
                                    self.align_reel_to_layout();
                                }
                                self.recenter_prefetch();
                                self.prefetch.dirty = true;
                                let max_start = self.reel_max_start();
                                if self.current > max_start {
                                    self.current = max_start;
                                }
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
                    },
                );
            });

            self.top_bar_rect = Some(top_panel.response.rect);

            let bottom_panel = egui::TopBottomPanel::bottom("stats").show(ctx, |ui| {
                ui.with_layer_id(
                    egui::LayerId::new(egui::Order::Foreground, egui::Id::new("stats_widgets")),
                    |ui| {
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
                    },
                );
            });

            self.bottom_bar_rect = Some(bottom_panel.response.rect);
        } else {
            self.top_bar_rect = None;
            self.bottom_bar_rect = None;
        }

        // 6) Central panel — reel (carousel) or single-image with crossfade
        egui::CentralPanel::default().show(ctx, |ui| {
            use egui::{Color32, Pos2, Rect, Sense};
            if self.images.is_empty() {
                return;
            }

            let mut context_actions: Vec<ImageMenuAction> = Vec::new();

            let avail = ui.available_rect_before_wrap();
            let viewport = avail.size();
            let visible_now = self.visible_per_page().min(self.images.len()).max(1);
            let multi_page = !self.reel_enabled && visible_now > 1;
            let pointer_down = input.pointer.any_down();
            let suppress_drag = self.suppress_drag_once || self.suppress_drag_until_release;

            if let Some(p) = input.pointer.hover_pos() {
                self.last_cursor = Some(p);
            }
            let cursor = self.last_cursor.unwrap_or(avail.center());

            // Mouse-side-button zoom
            if !self.reel_enabled {
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

            ui.set_clip_rect(avail);

            let painter = ui.painter().with_clip_rect(avail);

            if self.reel_enabled {
                // Recompute reel animation before drawing the requested page size.
                self.step_reel_animation(ctx, viewport);

                let total = self.images.len();
                if total == 0 {
                    return;
                }
                if viewport.x <= 0.0 || viewport.y <= 0.0 {
                    return;
                }


                if visible_now == 1 {
                    if self.reel_looping {
                        let raw_pos = self.reel_pos;
                        let base_floor = raw_pos.floor();
                        let frac = raw_pos - base_floor;
                        let base_global = base_floor as i64;

                        let mut tile_cache: HashMap<i64, (Vec2, egui::TextureId)> = HashMap::new();
                        let mut tile_for = |g_idx: i64| -> (Vec2, egui::TextureId) {
                            if let Some(val) = tile_cache.get(&g_idx) {
                                *val
                            } else {
                                let idx = wrap_index(g_idx, total);
                                let entry = &self.images[idx];
                                let fit = (viewport.x / entry.tex.size_vec2().x)
                                    .min(viewport.y / entry.tex.size_vec2().y)
                                    .min(1.0);
                                let data = (entry.tex.size_vec2() * fit * self.zoom, entry.tex.id());
                                tile_cache.insert(g_idx, data);
                                data
                            }
                        };

                        let mut base_center_x = avail.center().x + self.pan.x;
                        if frac > 0.0 {
                            let (size_a, _) = tile_for(base_global);
                            let (size_b, _) = tile_for(base_global + 1);
                            let step = 0.5 * (size_a.x + size_b.x);
                            base_center_x -= frac * step;
                        }

                        let mut drag_dx = 0.0f32;
                        let mut drag_dy = 0.0f32;
                        let allow_drag = pointer_down && !suppress_drag;

                        for k in (-REEL_NEIGHBORS..=REEL_NEIGHBORS).rev() {
                            let global_idx = base_global + k as i64;
                            let (size, tex_id) = tile_for(global_idx);
                            let idx_for_menu = wrap_index(global_idx, total);
                            let menu_path = self.images[idx_for_menu].path.clone();

                            let mut cx = base_center_x;
                            if k < 0 {
                                let mut g = base_global;
                                while g > global_idx {
                                    let (size_prev, _) = tile_for(g - 1);
                                    let (size_curr, _) = tile_for(g);
                                    cx -= 0.5 * (size_prev.x + size_curr.x);
                                    g -= 1;
                                }
                            } else if k > 0 {
                                let mut g = base_global;
                                while g < global_idx {
                                    let (size_a, _) = tile_for(g);
                                    let (size_b, _) = tile_for(g + 1);
                                    cx += 0.5 * (size_a.x + size_b.x);
                                    g += 1;
                                }
                            }

                            let cy = avail.center().y + self.pan.y;
                            let rect = Rect::from_center_size(Pos2::new(cx, cy), size);

                            let resp = ui.allocate_rect(rect, Sense::click_and_drag());
                            painter.image(
                                tex_id,
                                rect,
                                egui::Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                                Color32::WHITE,
                            );

                            if resp.hovered() {
                                set_grab(false);
                            }
                            if resp.dragged() && allow_drag {
                                let delta = resp.drag_delta();
                                drag_dx += delta.x;
                                drag_dy += delta.y;
                            }
                            if let Some(action) = ViewerApp::image_context_menu(&resp, menu_path) {
                                context_actions.push(action);
                            }
                        }

                        if allow_drag && (drag_dx != 0.0 || drag_dy != 0.0) {
                            let (size_a, _) = tile_for(base_global);
                            let (size_b, _) = tile_for(base_global + 1);
                            let mut step_ref = 0.5 * (size_a.x + size_b.x);
                            if !step_ref.is_finite() || step_ref <= 0.0 {
                                step_ref = 256.0;
                            }
                            self.reel_target -= drag_dx / step_ref;
                            self.reel_auto_phase = self.reel_target;
                            self.pan.y += drag_dy;
                            set_grab(true);
                        }
                    } else {
                        let max_start = self.reel_max_start() as f32;
                        let raw_pos = self.reel_pos.clamp(0.0, max_start);
                        let mut base_idx = raw_pos.floor() as usize;
                        if base_idx >= total {
                            base_idx = total - 1;
                        }
                        let frac = (raw_pos - base_idx as f32).clamp(0.0, 1.0);

                        let mut neighbor_indices = Vec::new();
                        for k in -REEL_NEIGHBORS..=REEL_NEIGHBORS {
                            let idx_isize = base_idx as isize + k;
                            if (0..(total as isize)).contains(&idx_isize) {
                                neighbor_indices.push(idx_isize as usize);
                            }
                        }

                        let mut tile_cache: HashMap<usize, (Vec2, egui::TextureId)> = HashMap::new();
                        for idx in &neighbor_indices {
                            let entry = &self.images[*idx];
                            let fit = (viewport.x / entry.tex.size_vec2().x)
                                .min(viewport.y / entry.tex.size_vec2().y)
                                .min(1.0);
                            tile_cache.insert(
                                *idx,
                                (entry.tex.size_vec2() * fit * self.zoom, entry.tex.id()),
                            );
                        }

                        let step_between = |a: usize, b: usize| -> f32 {
                            let &(size_a, _) = tile_cache.get(&a).expect("missing tile size");
                            let &(size_b, _) = tile_cache.get(&b).expect("missing tile size");
                            0.5 * (size_a.x + size_b.x)
                        };

                        let mut base_center_x = avail.center().x + self.pan.x;
                        if frac > 0.0 && base_idx + 1 < total {
                            base_center_x -= frac * step_between(base_idx, base_idx + 1);
                        }

                        let mut drag_dx = 0.0f32;
                        let mut drag_dy = 0.0f32;
                        let allow_drag = pointer_down && !suppress_drag;

                        for k in (-REEL_NEIGHBORS..=REEL_NEIGHBORS).rev() {
                            let idx_isize = base_idx as isize + k;
                            if !(0..(total as isize)).contains(&idx_isize) {
                                continue;
                            }
                            let idx = idx_isize as usize;
                            let menu_path = self.images[idx].path.clone();
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
                            painter.image(
                                tex_id,
                                rect,
                                egui::Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                                Color32::WHITE,
                            );

                            if resp.hovered() {
                                set_grab(false);
                            }
                            if resp.dragged() && allow_drag {
                                let delta = resp.drag_delta();
                                drag_dx += delta.x;
                                drag_dy += delta.y;
                            }
                            if let Some(action) = ViewerApp::image_context_menu(&resp, menu_path) {
                                context_actions.push(action);
                            }
                        }

                        if allow_drag && (drag_dx != 0.0 || drag_dy != 0.0) {
                            let step_ref = if base_idx + 1 < total {
                                step_between(base_idx, base_idx + 1).max(1.0)
                            } else if base_idx > 0 {
                                step_between(base_idx - 1, base_idx).max(1.0)
                            } else {
                                256.0
                            };
                            self.reel_target =
                                (self.reel_target - drag_dx / step_ref).clamp(0.0, max_start);
                            self.reel_auto_phase = self.reel_target;
                            self.pan.y += drag_dy;
                            set_grab(true);
                        }
                    }
                } else {
                    if self.reel_looping {
                        let visible = visible_now.min(total).max(1);
                        let gap = 0.0;
                        let gap_count = visible.saturating_sub(1) as f32;
                        let mut tile_width = (viewport.x - gap * gap_count).max(1.0) / visible as f32;
                        if !tile_width.is_finite() {
                            tile_width = viewport.x.max(1.0) / visible as f32;
                        }
                        let tile_step = tile_width + if visible > 1 { gap } else { 0.0 };
                        let total_span = tile_width * visible as f32 + gap * gap_count;

                        let raw_pos = self.reel_pos;
                        let start_floor = raw_pos.floor();
                        let mut offset = raw_pos - start_floor;
                        if !offset.is_finite() {
                            offset = 0.0;
                        }
                        offset = offset.fract();
                        if offset < 0.0 {
                            offset += 1.0;
                        }
                        let start_global = start_floor as i64;
                        let tiles_to_draw = visible + if offset > 0.0001 { 1 } else { 0 };

                        let base_left =
                            avail.center().x + self.pan.x - 0.5 * total_span - offset * tile_step;
                        let center_y = avail.center().y + self.pan.y;

                        let mut drag_dx = 0.0f32;
                        let mut drag_dy = 0.0f32;
                        let allow_drag = pointer_down && !suppress_drag;

                        for i in 0..tiles_to_draw {
                            let global_idx = start_global + i as i64;
                            let idx = wrap_index(global_idx, total);
                            let entry = &self.images[idx];
                            let menu_path = entry.path.clone();
                            let tex_size = entry.tex.size_vec2();
                            if tex_size.x <= 0.0 || tex_size.y <= 0.0 {
                                continue;
                            }

                            let fit_w = tile_width / tex_size.x;
                            let fit_h = viewport.y / tex_size.y;
                            let mut fit = fit_w.min(fit_h).min(1.0);
                            if !fit.is_finite() || fit <= 0.0 {
                                fit = 1.0;
                            }
                            let size = tex_size * fit * self.zoom;

                            let center_x = base_left + i as f32 * tile_step + tile_width * 0.5;
                            let rect = Rect::from_center_size(Pos2::new(center_x, center_y), size);

                            let resp = ui.allocate_rect(rect, Sense::click_and_drag());
                            painter.image(
                                entry.tex.id(),
                                rect,
                                egui::Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                                Color32::WHITE,
                            );

                            if resp.hovered() {
                                set_grab(false);
                            }
                            if resp.dragged() && allow_drag {
                                let delta = resp.drag_delta();
                                drag_dx += delta.x;
                                drag_dy += delta.y;
                            }
                            if let Some(action) = ViewerApp::image_context_menu(&resp, menu_path) {
                                context_actions.push(action);
                            }
                        }

                        if allow_drag && (drag_dx != 0.0 || drag_dy != 0.0) {
                            let step = tile_step.max(1.0);
                            self.reel_target -= drag_dx / step;
                            self.reel_auto_phase = self.reel_target;
                            self.pan.y += drag_dy;
                            set_grab(true);
                        }
                    } else {
                        let visible = visible_now.min(total).max(1);
                        let gap = 0.0;
                        let gap_count = visible.saturating_sub(1) as f32;
                        let mut tile_width = (viewport.x - gap * gap_count).max(1.0) / visible as f32;
                        if !tile_width.is_finite() {
                            tile_width = viewport.x.max(1.0) / visible as f32;
                        }
                        let tile_step = tile_width + if visible > 1 { gap } else { 0.0 };
                        let total_span = tile_width * visible as f32 + gap * gap_count;

                        let max_start = self.reel_max_start() as f32;
                        let raw_pos = self.reel_pos.clamp(0.0, max_start);
                        let mut start_idx = raw_pos.floor() as usize;
                        if start_idx + visible > total {
                            start_idx = total.saturating_sub(visible);
                        }
                        let mut offset = raw_pos - start_idx as f32;
                        if !offset.is_finite() || offset < 0.0 {
                            offset = 0.0;
                        }

                        let extra_tile = offset > 0.0001 && start_idx + visible < total;
                        let tiles_to_draw = visible + if extra_tile { 1 } else { 0 };

                        let base_left =
                            avail.center().x + self.pan.x - 0.5 * total_span - offset * tile_step;
                        let center_y = avail.center().y + self.pan.y;

                        let mut drag_dx = 0.0f32;
                        let mut drag_dy = 0.0f32;
                        let allow_drag = pointer_down && !suppress_drag;

                        for i in 0..tiles_to_draw {
                            let idx = start_idx + i;
                            if idx >= total {
                                break;
                            }
                            let entry = &self.images[idx];
                            let menu_path = entry.path.clone();
                            let tex_size = entry.tex.size_vec2();
                            if tex_size.x <= 0.0 || tex_size.y <= 0.0 {
                                continue;
                            }

                            let fit_w = tile_width / tex_size.x;
                            let fit_h = viewport.y / tex_size.y;
                            let mut fit = fit_w.min(fit_h).min(1.0);
                            if !fit.is_finite() || fit <= 0.0 {
                                fit = 1.0;
                            }
                            let size = tex_size * fit * self.zoom;

                            let center_x = base_left + i as f32 * tile_step + tile_width * 0.5;
                            let rect = Rect::from_center_size(Pos2::new(center_x, center_y), size);

                            let resp = ui.allocate_rect(rect, Sense::click_and_drag());
                            painter.image(
                                entry.tex.id(),
                                rect,
                                egui::Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                                Color32::WHITE,
                            );

                            if resp.hovered() {
                                set_grab(false);
                            }
                            if resp.dragged() && allow_drag {
                                let delta = resp.drag_delta();
                                drag_dx += delta.x;
                                drag_dy += delta.y;
                            }
                        if let Some(action) = ViewerApp::image_context_menu(&resp, menu_path) {
                            context_actions.push(action);
                        }
                        }

                        if allow_drag && (drag_dx != 0.0 || drag_dy != 0.0) {
                            let step = tile_step.max(1.0);
                            self.reel_target = (self.reel_target - drag_dx / step).clamp(0.0, max_start);
                            self.reel_auto_phase = self.reel_target;
                            self.pan.y += drag_dy;
                            set_grab(true);
                        }
                    }
                }
            } else if multi_page {
                let total = self.images.len();
                if total == 0 || viewport.x <= 0.0 || viewport.y <= 0.0 {
                    return;
                }

                let mut fade_prev: Option<(usize, f32)> = None;
                let mut fade_cur = 1.0f32;
                let mut clear_transition = false;
                if let Some(t) = self.transition.as_ref() {
                    let p = (if t.dur <= 0.0 { 1.0 } else { t.progress() }).min(1.0);
                    let a = Transition::smoothstep(p).clamp(0.0, 1.0);
                    if p < 1.0 {
                        ctx.request_repaint();
                    } else {
                        clear_transition = true;
                    }
                    fade_prev = Some((t.from_idx, (1.0 - a).clamp(0.0, 1.0)));
                    fade_cur = a;
                }

                let gap = REEL_GAP_PX;
                let gap_count = visible_now.saturating_sub(1) as f32;
                let mut tile_width = (viewport.x - gap * gap_count).max(1.0) / visible_now as f32;
                if !tile_width.is_finite() {
                    tile_width = viewport.x.max(1.0) / visible_now as f32;
                }
                let tile_step = tile_width + if visible_now > 1 { gap } else { 0.0 };
                let total_span = tile_width * visible_now as f32 + gap * gap_count;
                let base_left = avail.center().x + self.pan.x - 0.5 * total_span;
                let center_y = avail.center().y + self.pan.y;

                let draw_static_page = |start_idx: usize, tint: Color32| {
                    if tint.a() == 0 {
                        return;
                    }
                    for i in 0..visible_now {
                        let idx = (start_idx + i) % total;
                        let entry = &self.images[idx];
                        let tex_size = entry.tex.size_vec2();
                        if tex_size.x <= 0.0 || tex_size.y <= 0.0 {
                            continue;
                        }
                        let fit_w = tile_width / tex_size.x;
                        let fit_h = viewport.y / tex_size.y;
                        let mut fit = fit_w.min(fit_h).min(1.0);
                        if !fit.is_finite() || fit <= 0.0 {
                            fit = 1.0;
                        }
                        let size = tex_size * fit * self.zoom;
                        let center_x = base_left + i as f32 * tile_step + tile_width * 0.5;
                        let rect = Rect::from_center_size(Pos2::new(center_x, center_y), size);
                        painter.image(
                            entry.tex.id(),
                            rect,
                            egui::Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                            tint,
                        );
                    }
                };

                if let Some((prev_start, prev_alpha)) = fade_prev {
                    let tint = Color32::from_white_alpha(
                        (prev_alpha.clamp(0.0, 1.0) * 255.0).round() as u8,
                    );
                    draw_static_page(prev_start % total, tint);
                }

                let cur_tint =
                    Color32::from_white_alpha((fade_cur.clamp(0.0, 1.0) * 255.0).round() as u8);
                let start_idx = self.current % total;
                if cur_tint.a() != 0 {
                    for i in 0..visible_now {
                        let idx = (start_idx + i) % total;
                        let entry = &self.images[idx];
                        let menu_path = entry.path.clone();
                        let tex_size = entry.tex.size_vec2();
                        if tex_size.x <= 0.0 || tex_size.y <= 0.0 {
                            continue;
                        }
                        let fit_w = tile_width / tex_size.x;
                        let fit_h = viewport.y / tex_size.y;
                        let mut fit = fit_w.min(fit_h).min(1.0);
                        if !fit.is_finite() || fit <= 0.0 {
                            fit = 1.0;
                        }
                        let size = tex_size * fit * self.zoom;
                        let center_x = base_left + i as f32 * tile_step + tile_width * 0.5;
                        let rect = Rect::from_center_size(Pos2::new(center_x, center_y), size);
                        let resp = ui.allocate_rect(rect, Sense::click_and_drag());
                        painter.image(
                            entry.tex.id(),
                            rect,
                            egui::Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                            cur_tint,
                        );
                        if resp.hovered() {
                            set_grab(false);
                        }
                        if resp.dragged()
                            && !(self.suppress_drag_once || self.suppress_drag_until_release)
                        {
                            self.pan += resp.drag_delta();
                            set_grab(true);
                        }
                        if let Some(action) = ViewerApp::image_context_menu(&resp, menu_path) {
                            context_actions.push(action);
                        }
                    }
                }

                if clear_transition {
                    self.transition = None;
                }
            } else {
                // ── SINGLE IMAGE (with optional crossfade) ───────────────────
                let cur_idx = self.current;
                let cur_img = &self.images[cur_idx];
                let menu_path = cur_img.path.clone();
                let cur_fit = (avail.width() / cur_img.tex.size_vec2().x)
                    .min(avail.height() / cur_img.tex.size_vec2().y)
                    .min(1.0);
                let cur_size = cur_img.tex.size_vec2() * cur_fit * self.zoom;
                let cur_rect = Rect::from_center_size(avail.center() + self.pan, cur_size);

                // Base interaction rect
                let resp = ui.allocate_rect(cur_rect, Sense::click_and_drag());

                // Crossfade if active
                let mut drew_prev = false;
                if let Some(t) = &self.transition {
                    if t.from_idx < self.images.len() && t.to_idx < self.images.len() {
                        let p = (if t.dur <= 0.0 { 1.0 } else { t.progress() }).min(1.0);
                        let a = Transition::smoothstep(p);
                        let prev = &self.images[t.from_idx];

                        let prev_fit = (avail.width() / prev.tex.size_vec2().x)
                            .min(avail.height() / prev.tex.size_vec2().y)
                            .min(1.0);
                        let prev_size = prev.tex.size_vec2() * prev_fit * self.zoom;
                        let prev_rect =
                            Rect::from_center_size(avail.center() + self.pan, prev_size);

                        painter.image(
                            prev.tex.id(),
                            prev_rect,
                            egui::Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                            Color32::from_white_alpha(((1.0 - a) * 255.0) as u8),
                        );
                        drew_prev = true;

                        if p >= 1.0 {
                            self.transition = None;
                        } else {
                            ctx.request_repaint();
                        }
                    } else {
                        self.transition = None;
                    }
                }

                // Draw current
                painter.image(
                    cur_img.tex.id(),
                    cur_rect,
                    egui::Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                    if drew_prev {
                        if let Some(t) = &self.transition {
                            let p = (if t.dur <= 0.0 { 1.0 } else { t.progress() }).min(1.0);
                            let a = Transition::smoothstep(p);
                            Color32::from_white_alpha((a * 255.0) as u8)
                        } else {
                            Color32::from_white_alpha(255)
                        }
                    } else {
                        Color32::from_white_alpha(255)
                    },
                );

                // Panning
                if resp.hovered() {
                    set_grab(false);
                }
                if resp.dragged() {
                    if !(self.suppress_drag_once || self.suppress_drag_until_release) {
                        self.pan += resp.drag_delta();
                        set_grab(true);
                    }
                }
                if let Some(action) = ViewerApp::image_context_menu(&resp, menu_path) {
                    context_actions.push(action);
                }
            }

            ViewerApp::process_image_actions(context_actions);
        });

        if self.suppress_drag_once {
            self.suppress_drag_once = false;
        }
        if self.suppress_drag_until_release && !input.pointer.any_down() {
            self.suppress_drag_until_release = false;
        }

        // Drive reel when enabled (kept here so it runs even if not drawing every frame)
        if self.reel_enabled {
            // Use current viewport size for scale targeting if needed
            let viewport = ctx.input(|i| i.screen_rect().size());
            self.step_reel_animation(ctx, viewport);
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

fn wrap_index(idx: i64, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let len_i = len as i64;
    let mut r = idx % len_i;
    if r < 0 {
        r += len_i;
    }
    r as usize
}
