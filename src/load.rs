use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    sync::OnceLock,
    time::SystemTime,
};

use crossbeam_channel::{select, Receiver, Sender};
use eframe::egui;
use image::{codecs::gif::GifDecoder, AnimationDecoder, DynamicImage, ImageReader};

/* ───────────────────────── channel types / caps ─────────────────── */

/// RGBA frame data for animated images (e.g. GIFs).
pub struct AnimatedFrame {
    pub width: usize,
    pub height: usize,
    pub rgba: Vec<u8>,
    /// Frame delay in seconds.
    pub delay: f32,
}

/// Payload describing the decoded image data.
pub enum ImgPayload {
    Static { rgba: Vec<u8> },
    Animated {
        frames: Vec<AnimatedFrame>,
        loop_count: Option<u16>,
    },
}

/// Message emitted by decoder workers after loading an image.
pub struct ImgMsg {
    pub path: PathBuf,
    pub width: usize,
    pub height: usize,
    pub bytes: u64,
    pub created: Option<SystemTime>,
    pub payload: ImgPayload,
}
/// (generation_id, path) — workers drop jobs whose gen_id != current
pub type JobMsg = (u64, PathBuf);
/// discovered path
pub type PathMsg = PathBuf;

// Suggested capacities
pub const IMG_CHAN_CAP: usize = 1024;
pub const PATHS_CHAN_CAP: usize = 8192;

// Back-pressure for decode job queue (bump to keep pipeline full with many cores)
pub const MAX_ENQUEUED_JOBS: usize = 8192;

/// Clamp for extremely short/zero GIF frame delays (roughly 60 FPS).
const MIN_GIF_FRAME_DELAY_SEC: f32 = 0.016;

/* ───────────────────────── prefetch tuneables ───────────────────── */

/// Initial (fallback) radius; will be widened to full list on recenter().
pub const PREFETCH_RADIUS: usize = 512;

/// Legacy hint from the UI; we’ll adapt beyond this in `tick()`.
pub const PREFETCH_PER_TICK: usize = 64;

/* ───────────────────────── queue state (dedupe) ─────────────────── */

/// Tracks which paths we've already uploaded (seen) and which are enqueued for decode.
#[derive(Default)]
pub struct QueueState {
    pub seen: HashSet<PathBuf>,
    pub enqueued: HashSet<PathBuf>,
}

impl QueueState {
    #[inline]
    pub fn clear(&mut self) {
        self.seen.clear();
        self.enqueued.clear();
    }

    /// Mark a decoded path as seen (after texture upload, etc.).
    #[inline]
    pub fn mark_seen(&mut self, path: &PathBuf) {
        self.seen.insert(path.clone());
        // Once seen, it's no longer considered "enqueued only".
        self.enqueued.remove(path);
    }

    /// Attempt to enqueue `path` if it's neither seen nor already enqueued.
    /// Returns true if enqueued.
    #[inline]
    pub fn try_enqueue_unique(
        &mut self,
        job_tx: &Sender<JobMsg>,
        gen_id: u64,
        path: PathBuf,
    ) -> bool {
        if self.seen.contains(&path) || self.enqueued.contains(&path) {
            return false;
        }
        if job_tx.try_send((gen_id, path.clone())).is_ok() {
            self.enqueued.insert(path);
            true
        } else {
            false
        }
    }
}

/* ───────────────────────── prefetch scheduler ───────────────────── */

/// Minimal, GUI-agnostic prefetch state machine (sliding window).
#[derive(Debug, Clone)]
pub struct Prefetcher {
    pub center_idx: usize,
    pub step: isize,
    pub dirty: bool,
    pub radius: usize,
}

impl Default for Prefetcher {
    fn default() -> Self {
        Self {
            center_idx: 0,
            step: 0,
            dirty: true,
            radius: PREFETCH_RADIUS,
        }
    }
}

impl Prefetcher {
    #[inline]
    pub fn new(radius: usize) -> Self {
        Self {
            radius,
            ..Default::default()
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        self.center_idx = 0;
        self.step = 0;
        self.dirty = true;
    }

    /// Recenter around a known index (computed by the GUI or caller).
    #[inline]
    pub fn recenter_to(&mut self, idx: usize) {
        self.center_idx = idx;
        self.step = 0;
        self.dirty = true;
    }

    /// Convenience: derive center index from either the current path or a pending target name,
    /// falling back to the first file. Also widen radius to cover the whole list so we’ll enqueue
    /// EVERYTHING, expanding from the center.
    pub fn recenter(
        &mut self,
        files: &[PathBuf],
        current_path: Option<&PathBuf>,
        pending_target_name: Option<&str>,
        index_of: &HashMap<PathBuf, usize>,
    ) {
        let idx = if let Some(cp) = current_path {
            index_of.get(cp).copied()
        } else if let Some(name) = pending_target_name {
            files.iter().position(|p| {
                p.file_name()
                    .and_then(|s| s.to_str())
                    .map(|s| s == name)
                    .unwrap_or(false)
            })
        } else {
            Some(0)
        }
        .unwrap_or(0);

        // Cover the whole list — smooth, complete loading outward from the center
        self.radius = files.len().saturating_sub(1);

        self.recenter_to(idx);
    }

    /// Adaptive per-tick budget based on CPU and queue headroom.
    #[inline]
    fn adaptive_budget(
        job_tx: &Sender<JobMsg>,
        max_enqueued_jobs: usize,
        ui_hint: usize,
        seen: usize,
    ) -> usize {
        let free = max_enqueued_jobs.saturating_sub(job_tx.len());
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        // Early phase: be more aggressive until a few hundred are enqueued/seen,
        // then settle. Aim to keep ~16*cores queued ahead but never exceed headroom.
        let ramp = if seen < 256 { 2 } else { 1 };
        let target = cores.saturating_mul(16 * ramp);

        // Allow bursts early; never exceed remaining channel capacity.
        free.min(target.max(ui_hint).min(1024))
    }

    /// Walk outward from `center_idx` in the sequence: 0, +1, -1, +2, -2, ...
    /// Enqueue new decode jobs into `job_tx`, respecting `qstate`.
    ///
    /// Returns the number of paths newly enqueued this tick.
    pub fn tick(
        &mut self,
        files: &[PathBuf],
        qstate: &mut QueueState,
        job_tx: &Sender<JobMsg>,
        max_to_enqueue_hint: usize,
        max_enqueued_jobs: usize,
        gen_id: u64,
    ) -> usize {
        if !self.dirty || files.is_empty() {
            return 0;
        }
        if job_tx.len() >= max_enqueued_jobs {
            return 0;
        }

        let budget = Self::adaptive_budget(
            job_tx,
            max_enqueued_jobs,
            max_to_enqueue_hint,
            qstate.seen.len(),
        );
        if budget == 0 {
            return 0;
        }

        let mut emitted = 0usize;
        while emitted < budget {
            let step_abs = self.step.unsigned_abs();
            if step_abs > self.radius {
                break;
            }

            let idx = if self.step == 0 {
                self.center_idx as isize
            } else {
                self.center_idx as isize + self.step
            };

            // Next step: 0 → +1 → -1 → +2 → -2 → ...
            self.step = if self.step <= 0 {
                -self.step + 1
            } else {
                -self.step
            };

            if idx < 0 || (idx as usize) >= files.len() {
                continue;
            }
            let path = &files[idx as usize];

            if qstate.seen.contains(path) || qstate.enqueued.contains(path) {
                continue;
            }
            if job_tx.len() >= max_enqueued_jobs {
                break;
            }
            if qstate.try_enqueue_unique(job_tx, gen_id, path.clone()) {
                emitted += 1;
            }
        }

        if self.step.unsigned_abs() > self.radius {
            self.dirty = false;
        }
        emitted
    }
}

/* ───────────────────────── decoding / workers ───────────────────── */

#[inline]
fn suggested_decoder_threads() -> usize {
    // Use ~3/4 of logical cores for decoders (avoid starving GUI + IO),
    // clamp conservatively for big workstations.
    let logical = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let target = (logical * 3) / 4;
    target.clamp(3, 24)
}

#[cfg(windows)]
fn open_file_sequential(path: &PathBuf) -> std::io::Result<std::fs::File> {
    use std::fs::OpenOptions;
    use std::os::windows::fs::OpenOptionsExt;
    // FILE_FLAG_SEQUENTIAL_SCAN
    const SEQ: u32 = 0x0800_0000;
    OpenOptions::new().read(true).custom_flags(SEQ).open(path)
}

#[cfg(windows)]
fn try_decode_with_hint(path: &PathBuf) -> Result<DynamicImage, ()> {
    use std::io::BufReader;
    let f = open_file_sequential(path).map_err(|_| ())?;
    let fmt = image::ImageFormat::from_path(path).map_err(|_| ())?;
    ImageReader::with_format(BufReader::new(f), fmt)
        .decode()
        .map_err(|_| ())
}

#[cfg(not(windows))]
fn try_decode_with_hint(_path: &PathBuf) -> Result<DynamicImage, ()> {
    Err(())
}

fn decode_image(path: &PathBuf) -> Result<DynamicImage, ()> {
    // Synchronous file IO, decode on CPU (leverages SIMD in image >= 0.25).
    try_decode_with_hint(path).or_else(|_| {
        ImageReader::open(path)
            .map_err(|_| ())?
            .decode()
            .map_err(|_| ())
    })
}

#[inline]
fn file_metadata(path: &PathBuf) -> (u64, Option<SystemTime>) {
    match std::fs::metadata(path) {
        Ok(meta) => (meta.len(), meta.created().ok()),
        Err(_) => (0, None),
    }
}

#[inline]
fn scaled_dimensions(w: u32, h: u32, max_dim: u32) -> (u32, u32) {
    if w >= h {
        let ratio = max_dim as f32 / w as f32;
        let nh = ((h as f32) * ratio).round().max(1.0) as u32;
        (max_dim, nh.max(1))
    } else {
        let ratio = max_dim as f32 / h as f32;
        let nw = ((w as f32) * ratio).round().max(1.0) as u32;
        (nw.max(1), max_dim)
    }
}

fn decode_static_rgba(path: &PathBuf, max_dim: Option<u32>) -> Result<ImgMsg, ()> {
    use image::imageops::FilterType;

    let mut img = decode_image(path)?;
    if let Some(limit) = max_dim {
        let (w, h) = (img.width(), img.height());
        if w.max(h) > limit {
            let (nw, nh) = scaled_dimensions(w, h, limit);
            img = img.resize(nw, nh, FilterType::Triangle);
        }
    }

    let rgba = img.to_rgba8();
    let width = rgba.width() as usize;
    let height = rgba.height() as usize;
    let (bytes, created) = file_metadata(path);

    Ok(ImgMsg {
        path: path.clone(),
        width,
        height,
        bytes,
        created,
        payload: ImgPayload::Static { rgba: rgba.into_raw() },
    })
}

fn decode_gif_rgba(path: &PathBuf, max_dim: Option<u32>) -> Result<ImgMsg, ()> {
    use image::imageops::FilterType;
    use std::io::BufReader;

    #[cfg(windows)]
    let file = open_file_sequential(path).map_err(|_| ())?;
    #[cfg(not(windows))]
    let file = std::fs::File::open(path).map_err(|_| ())?;

    let decoder = GifDecoder::new(BufReader::new(file)).map_err(|_| ())?;
    let mut frames_iter = decoder.into_frames();
    let mut frames = Vec::new();

    while let Some(frame_result) = frames_iter.next() {
        let frame = frame_result.map_err(|_| ())?;
        let delay = frame.delay();
        let (numer, denom) = delay.numer_denom_ms();
        let mut img = DynamicImage::ImageRgba8(frame.into_buffer());

        if let Some(limit) = max_dim {
            let (w, h) = (img.width(), img.height());
            if w.max(h) > limit {
                let (nw, nh) = scaled_dimensions(w, h, limit);
                img = img.resize(nw, nh, FilterType::Triangle);
            }
        }

        let rgba = img.to_rgba8();
        let width = rgba.width() as usize;
        let height = rgba.height() as usize;

        let delay_ms = if denom == 0 {
            MIN_GIF_FRAME_DELAY_SEC * 1000.0
        } else {
            numer as f32 / denom as f32
        };
        let delay_sec = (delay_ms / 1000.0).max(MIN_GIF_FRAME_DELAY_SEC);

        frames.push(AnimatedFrame {
            width,
            height,
            rgba: rgba.into_raw(),
            delay: delay_sec,
        });
    }

    if frames.is_empty() {
        return Err(());
    }

    let (bytes, created) = file_metadata(path);
    let width = frames[0].width;
    let height = frames[0].height;

    Ok(ImgMsg {
        path: path.clone(),
        width,
        height,
        bytes,
        created,
        payload: ImgPayload::Animated {
            frames,
            loop_count: None,
        },
    })
}

fn decode_path_to_msg(path: &PathBuf, max_dim: Option<u32>) -> Result<ImgMsg, ()> {
    let is_gif = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("gif"))
        .unwrap_or(false);

    if is_gif {
        decode_gif_rgba(path, max_dim)
    } else {
        decode_static_rgba(path, max_dim)
    }
}

/* Global Rayon decoder pool (one-time init) */
static DECODER_POOL_INIT: OnceLock<()> = OnceLock::new();

#[inline]
fn init_decoder_pool(threads: usize) {
    DECODER_POOL_INIT.get_or_init(|| {
        let effective = if threads == 0 {
            suggested_decoder_threads()
        } else {
            threads
        };
        // Build the *global* pool once; lower OS priority for decoder threads on Windows.
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(effective.max(2))
            .thread_name(|i| format!("decoder-{}", i))
            .start_handler(|_| {
                #[cfg(windows)]
                unsafe {
                    use winapi::um::processthreadsapi::{GetCurrentThread, SetThreadPriority};
                    use winapi::um::winbase::THREAD_PRIORITY_BELOW_NORMAL;
                    let _ =
                        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL as i32);
                }
                // #[cfg(unix)]
                // unsafe { libc::nice(10); } // optional
            })
            .build_global();
    });
}

pub fn start_decoder_workers_pool(
    job_rx: Receiver<JobMsg>,
    prio_job_rx: Option<Receiver<JobMsg>>,
    img_tx: Sender<ImgMsg>,
    egui_ctx: egui::Context,
    current_gen: std::sync::Arc<std::sync::atomic::AtomicU64>,
    threads: usize,
    use_downscaled: bool,
    max_dim: u32,
) {
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    // Ensure the global Rayon pool exists
    init_decoder_pool(threads);
    let effective = if threads == 0 {
        suggested_decoder_threads()
    } else {
        threads
    };

    // Spawn long-lived workers onto the *global* pool
    for _ in 0..effective.max(2) {
        let rx = job_rx.clone();
        let prx = prio_job_rx.clone();
        let tx = img_tx.clone();
        let ctx = egui_ctx.clone();
        let gen = current_gen.clone();

        rayon::spawn(move || {
            loop {
                // Steady frame cadence ≈60 FPS so scrolling animation is smooth
                ctx.request_repaint_after(Duration::from_millis(16));

                // Prefer priority queue when available; otherwise fairly receive either.
                let job = if let Some(ref pr) = prx {
                    select! {
                        recv(pr) -> r => r.ok(),
                        recv(rx) -> r => r.ok(),
                        default => {
                            match rx.recv() { Ok(j) => Some(j), Err(_) => None }
                        }
                    }
                } else {
                    match rx.recv() {
                        Ok(j) => Some(j),
                        Err(_) => None,
                    }
                };

                let Some((job_gen, path)) = job else { break };

                // Drop stale work (generation bumped on new loads)
                if job_gen != gen.load(Ordering::Relaxed) {
                    continue;
                }

                let decoded = decode_path_to_msg(&path, use_downscaled.then_some(max_dim));

                if let Ok(msg) = decoded {
                    if job_gen != gen.load(Ordering::Relaxed) {
                        continue;
                    }
                    let _ = tx.send(msg);
                    // Gentle nudge for prompt rendering without spamming
                    ctx.request_repaint_after(Duration::from_millis(8));
                }

                std::thread::yield_now();
            }
        });
    }
}

/* ───────────────────────── filesystem enumeration ───────────────── */

pub fn enumerate_paths(dir: PathBuf, paths_tx: Sender<PathMsg>, egui_ctx: egui::Context) {
    use jwalk::{Parallelism, WalkDir};
    use std::ffi::OsStr;
    use std::thread::{sleep, yield_now};
    use std::time::Duration;

    #[inline]
    fn is_img_ext(path: &PathBuf) -> bool {
        match path
            .extension()
            .and_then(OsStr::to_str)
            .map(|s| s.to_ascii_lowercase())
        {
            Some(ref e)
                if matches!(
                    e.as_str(),
                    "png" | "jpg" | "jpeg" | "bmp" | "gif" | "tiff" | "webp"
                ) =>
            {
                true
            }
            _ => false,
        }
    }

    // Keep walker modest to avoid stealing CPU from decoders and the GUI.
    let logical = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let walker_threads = logical.saturating_div(2).clamp(1, 4);

    // Coalesced, rate-limited dispatch:
    // - send in batches of 256
    // - short sleep after each send to let GUI breathe
    // - if the GUI path channel is near full, pause until it drains a bit
    const BATCH: usize = 256;
    const SLEEP_MS: u64 = 3;

    let (batch_tx, batch_rx) = crossbeam_channel::bounded::<Vec<PathBuf>>(walker_threads * 2);
    std::thread::spawn(move || {
        let mut batch = Vec::with_capacity(BATCH);

        WalkDir::new(dir)
            .follow_links(false)
            .sort(false)
            .skip_hidden(true)
            .parallelism(Parallelism::RayonNewPool(walker_threads))
            .into_iter()
            .for_each(|entry| {
                if let Ok(e) = entry {
                    if e.file_type().is_file() {
                        let p: PathBuf = e.path().to_path_buf();
                        if is_img_ext(&p) {
                            batch.push(p);
                            if batch.len() >= BATCH {
                                let mut out = Vec::new();
                                std::mem::swap(&mut out, &mut batch);
                                let _ = batch_tx.send(out);
                            }
                        }
                    }
                }
            });

        if !batch.is_empty() {
            let _ = batch_tx.send(batch);
        }
        // channel closed when thread exits
    });

    // Drain batches into the GUI channel with gentle pacing
    let mut total = 0usize;
    while let Ok(mut batch) = batch_rx.recv() {
        total += batch.len();

        // If GUI channel is nearly full, wait a bit to avoid stalling the UI thread
        while paths_tx.len() > PATHS_CHAN_CAP.saturating_sub(BATCH) {
            sleep(Duration::from_millis(SLEEP_MS));
        }

        for p in batch.drain(..) {
            let _ = paths_tx.send(p);
        }

        // Brief yield to give egui a frame
        yield_now();
        sleep(Duration::from_millis(SLEEP_MS));

        if total % 256 == 0 {
            egui_ctx.request_repaint();
        }
    }
    egui_ctx.request_repaint();
}

/* ───────────────────────── optional: channel factory ────────────── */

/// Convenience for keeping channel creation out of the UI layer.
#[allow(dead_code)]
pub fn new_channels() -> (
    Sender<ImgMsg>,
    Receiver<ImgMsg>,
    Sender<PathMsg>,
    Receiver<PathMsg>,
    Sender<JobMsg>,
    Receiver<JobMsg>,
) {
    use crossbeam_channel::bounded;
    let (img_tx, img_rx) = bounded::<ImgMsg>(IMG_CHAN_CAP);
    let (paths_tx, paths_rx) = bounded::<PathMsg>(PATHS_CHAN_CAP);
    let (job_tx, job_rx) = bounded::<JobMsg>(MAX_ENQUEUED_JOBS);
    (img_tx, img_rx, paths_tx, paths_rx, job_tx, job_rx)
}
