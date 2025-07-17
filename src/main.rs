use std::{path::PathBuf,time::SystemTime};
use crossbeam_channel::{unbounded,Receiver,Sender};
use eframe::{egui, egui::TextureHandle};
use egui::Vec2;
use image::{DynamicImage,ImageReader};
use rayon::prelude::*;
use rfd::FileDialog;
use walkdir::WalkDir;
type ImgMsg=(PathBuf,DynamicImage);
fn main()->eframe::Result<()>{
    eframe::run_native("Rust Multicore Image Viewer",eframe::NativeOptions::default(),Box::new(|_|Box::<ViewerApp>::default()))
}
#[derive(Clone,Copy,PartialEq,Eq)]enum SortKey{Name,Created,Size,Height,Width,Type}
impl SortKey{
    fn all()->[Self;6]{[Self::Name,Self::Created,Self::Size,Self::Height,Self::Width,Self::Type]}
    fn label(self)->&'static str{match self{
        SortKey::Name=>"Name",SortKey::Created=>"Date",SortKey::Size=>"Size",
        SortKey::Height=>"Height",SortKey::Width=>"Width",SortKey::Type=>"Type",
    }}
}
#[derive(Clone,Copy,PartialEq,Eq)]enum Layout{One,Two,Three}
impl Layout{
    fn all()->[Self;3]{[Self::One,Self::Two,Self::Three]}
    fn label(self)->&'static str{match self{
        Layout::One=>"1 image",Layout::Two=>"2 images",Layout::Three=>"3 images",
    }}
}
struct ImgEntry{name:String,ext:String,tex:TextureHandle,w:u32,h:u32,bytes:u64,created:Option<SystemTime>}
struct ViewerApp {
    images: Vec<ImgEntry>,
    current: usize,
    rx: Receiver<ImgMsg>,
    current_dir: Option<PathBuf>,
    is_fullscreen: bool,
    sort_key: SortKey,
    ascending: bool,
    layout: Layout,
    show_top_bar: bool,
    prev_alt_down: bool,
    zoom: f32, 
    pan: Vec2,
    last_zoom_time: f64,
    last_cursor: Option<egui::Pos2>,
    zoomed_once: bool,
}

impl Default for ViewerApp {
    fn default() -> Self {
        let (_tx, rx) = unbounded();
        Self {
            images: Vec::new(),
            current: 0,
            rx,
            current_dir: None,
            is_fullscreen: false,
            sort_key: SortKey::Name,
            ascending: true,
            layout: Layout::One,
            show_top_bar: true,
            prev_alt_down: false,
            zoom: 1.0,
            pan: Vec2::ZERO,
            last_zoom_time: 0.0,
            last_cursor: None,
            zoomed_once: false,
        }
    }
}

impl eframe::App for ViewerApp {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        use std::fs;
        use egui::{PointerButton, Key, ViewportCommand, Vec2, Rect, Sense, Pos2, Color32, ColorImage};
        use rfd::FileDialog;

        let input = ctx.input(|i| i.clone());
        let now   = input.time;

        // toggle both bars with Alt
        let alt = input.modifiers.alt;
        if alt && !self.prev_alt_down {
            self.show_top_bar = !self.show_top_bar;
        }
        self.prev_alt_down = alt;

        // load images
        while let Ok((path, img)) = self.rx.try_recv() {
            let (w, h) = (img.width(), img.height());
            let tex = ctx.load_texture(
                path.file_name().unwrap().to_string_lossy(),
                ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &img.into_rgba8()),
                egui::TextureOptions::default(),
            );
            let (bytes, created) = fs::metadata(&path)
                .map(|m| (m.len(), m.created().ok()))
                .unwrap_or((0, None));
            self.images.push(ImgEntry {
                name: path.file_name().unwrap().to_string_lossy().into(),
                ext:  path.extension().and_then(|s| s.to_str()).unwrap_or("").to_owned(),
                tex, w, h, bytes, created,
            });
            self.sort_images();
        }

        // keyboard & scroll nav
        if input.key_pressed(Key::ArrowRight) { self.next() }
        if input.key_pressed(Key::ArrowLeft)  { self.prev() }
        if input.key_pressed(Key::F11) {
            self.is_fullscreen = !self.is_fullscreen;
            ctx.send_viewport_cmd(ViewportCommand::Fullscreen(self.is_fullscreen));
        }
        match input.raw_scroll_delta.y {
            d if d >  0.0 => self.prev(),
            d if d <  0.0 => self.next(),
            _             => {}
        }

        // main image area
        egui::CentralPanel::default().show(ctx, |ui| {
            if !self.has_images() {
                ui.centered_and_justified(|ui| ui.label("No image loaded"));
                return;
            }
            let full = ui.available_rect_before_wrap();
            let bar_h = if self.show_top_bar { 32.0 } else { 0.0 };
            let avail = Rect::from_min_max(
                Pos2::new(full.min.x, full.min.y + bar_h),
                Pos2::new(full.max.x, full.max.y - bar_h),
            );

            if let Some(pos) = input.pointer.hover_pos() {
                self.last_cursor = Some(pos);
            }
            let cursor = self.last_cursor.unwrap_or(avail.center());

            // zoom buttons
            let zin        = input.pointer.button_down(PointerButton::Extra2);
            let zout       = input.pointer.button_down(PointerButton::Extra1);
            let zin_press  = input.pointer.button_pressed(PointerButton::Extra2);
            let zout_press = input.pointer.button_pressed(PointerButton::Extra1);

            if zin || zout {
                ctx.request_repaint();

                if zin_press || zout_press {
                    // 1) immediate zoom on press
                    let old    = self.zoom;
                    let factor = if zin_press { 1.1 } else { 1.0 / 1.1 };
                    let new    = (old * factor).clamp(0.1, 10.0);
                    self.pan  += (cursor - (avail.center() + self.pan)) * (1.0 - new / old);
                    self.zoom  = new;
                    self.last_zoom_time = now;
                    self.zoomed_once    = false;

                } else if !self.zoomed_once && now - self.last_zoom_time >= 0.3 {
                    // 2) second zoom after 0.3 s hold
                    let old    = self.zoom;
                    let factor = if zin { 1.1 } else { 1.0 / 1.1 };
                    let new    = (old * factor).clamp(0.1, 10.0);
                    self.pan  += (cursor - (avail.center() + self.pan)) * (1.0 - new / old);
                    self.zoom  = new;
                    self.last_zoom_time = now;
                    self.zoomed_once    = true;

                } else if self.zoomed_once && now - self.last_zoom_time >= 0.1 {
                    // 3) continuous zoom every 0.1 s thereafter
                    let old    = self.zoom;
                    let factor = if zin { 1.1 } else { 1.0 / 1.1 };
                    let new    = (old * factor).clamp(0.1, 10.0);
                    self.pan  += (cursor - (avail.center() + self.pan)) * (1.0 - new / old);
                    self.zoom  = new;
                    self.last_zoom_time = now;
                }
            } else {
                self.zoomed_once = false;
            }

            // draw image(s)
            match self.layout {
                Layout::One => {
                    let img  = self.current_img().unwrap();
                    let base = img.tex.size_vec2();
                    let fit  = (avail.width() / base.x)
                             .min(avail.height() / base.y)
                             .min(1.0);
                    let size = base * fit * self.zoom;
                    let rect = Rect::from_center_size(avail.center() + self.pan, size);
                    let resp = ui.allocate_rect(rect, Sense::drag());
                    ui.painter().image(
                        img.tex.id(),
                        rect,
                        Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0)),
                        Color32::WHITE,
                    );
                    if resp.dragged_by(PointerButton::Primary)
                        || resp.dragged_by(PointerButton::Secondary)
                        || resp.dragged_by(PointerButton::Middle)
                    {
                        self.pan += resp.drag_delta();
                    }
                }
                Layout::Two | Layout::Three => {
                    let cnt = if self.layout == Layout::Two { 2 } else { 3 };
                    let mut items = Vec::with_capacity(cnt);
                    for i in 0..cnt {
                        let img  = &self.images[(self.current + i) % self.images.len()];
                        let base = img.tex.size_vec2();
                        let fit  = (avail.width() / (cnt as f32) / base.x)
                                   .min(avail.height() / base.y)
                                   .min(1.0);
                        items.push((img.tex.id(), base * fit * self.zoom));
                    }
                    let total_w: f32 = items.iter().map(|(_, s)| s.x).sum();
                    let total_h: f32 = items.iter().map(|(_, s)| s.y).fold(0.0, f32::max);
                    let rect = Rect::from_center_size(avail.center() + self.pan, Vec2::new(total_w, total_h));
                    let resp = ui.allocate_rect(rect, Sense::drag());
                    ui.allocate_ui_at_rect(rect, |ui| {
                        ui.horizontal_centered(|ui| {
                            for (id, sz) in items {
                                ui.image((id, sz));
                            }
                        });
                    });
                    if resp.dragged_by(PointerButton::Primary)
                        || resp.dragged_by(PointerButton::Secondary)
                        || resp.dragged_by(PointerButton::Middle)
                    {
                        self.pan += resp.drag_delta();
                    }
                }
            }
        });

        // top toolbar (controls only)
        if self.show_top_bar {
            egui::TopBottomPanel::top("menu").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Open folder…").clicked() {
                        if let Some(dir) = FileDialog::new().pick_folder() {
                            self.spawn_loader(dir)
                        }
                    }
                    ui.separator();

                    // capture previous state
                    let prev_layout                = self.layout;
                    let (prev_sort_key, prev_ascending) = (self.sort_key, self.ascending);

                    ui.label("View:");
                    egui::ComboBox::from_id_source("layout")
                        .selected_text(self.layout.label())
                        .show_ui(ui, |ui| {
                            for l in Layout::all() {
                                ui.selectable_value(&mut self.layout, l, l.label());
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
                    egui::ComboBox::from_id_source("sort_order")
                        .selected_text(if self.ascending { "Asc" } else { "Desc" })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.ascending, true, "Asc");
                            ui.selectable_value(&mut self.ascending, false, "Desc");
                        });

                    // apply resets
                    if prev_layout != self.layout {
                        self.zoom        = 1.0;
                        self.pan         = Vec2::ZERO;
                        self.last_cursor = None;
                        self.zoomed_once = false;
                    }
                    if prev_sort_key != self.sort_key || prev_ascending != self.ascending {
                        self.sort_images();
                        self.zoom        = 1.0;
                        self.pan         = Vec2::ZERO;
                        self.last_cursor = None;
                        self.zoomed_once = false;
                    }

                    ui.separator();
                    ui.add_enabled(self.has_images(), egui::Button::new("◀ Prev"))
                        .clicked().then(|| self.prev());
                    ui.add_enabled(self.has_images(), egui::Button::new("Next ▶"))
                        .clicked().then(|| self.next());
                });
            });
        }

        // bottom stats bar
        if self.show_top_bar {
            egui::TopBottomPanel::bottom("stats").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if let Some(dir) = &self.current_dir {
                        ui.label(dir.display().to_string());
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
    }
}



impl ViewerApp {
    fn sort_images(&mut self) {
        let (asc, key) = (self.ascending, self.sort_key);
        self.images.sort_by(|a, b| {
            let ord = match key {
                SortKey::Name    => a.name.cmp(&b.name),
                SortKey::Created => a.created.cmp(&b.created),
                SortKey::Size    => a.bytes.cmp(&b.bytes),
                SortKey::Height  => a.h.cmp(&b.h).then(a.name.cmp(&b.name)),
                SortKey::Width   => a.w.cmp(&b.w).then(a.name.cmp(&b.name)),
                SortKey::Type    => a.ext.cmp(&b.ext).then(a.name.cmp(&b.name)),
            };
            if asc { ord } else { ord.reverse() }
        });
        self.current = self.current.min(self.images.len().saturating_sub(1));
    }

    fn spawn_loader(&mut self, dir: PathBuf) {
        self.images.clear();
        self.current = 0;
        let (tx, rx) = unbounded::<ImgMsg>();
        self.rx = rx;
        self.current_dir = Some(dir.clone());
        rayon::spawn(move || load_directory(dir, tx));
    }

    #[inline]
    fn has_images(&self) -> bool {
        !self.images.is_empty()
    }

    #[inline]
    fn current_img(&self) -> Option<&ImgEntry> {
        self.images.get(self.current)
    }

    fn next(&mut self) {
        if self.has_images() {
            self.current = (self.current + 1) % self.images.len();
            // reset pan & zoom
            self.zoom        = 1.0;
            self.pan         = Vec2::ZERO;
            self.last_cursor = None;
            self.zoomed_once = false;
        }
    }

    fn prev(&mut self) {
        if self.has_images() {
            self.current = (self.current + self.images.len() - 1) % self.images.len();
            // reset pan & zoom
            self.zoom        = 1.0;
            self.pan         = Vec2::ZERO;
            self.last_cursor = None;
            self.zoomed_once = false;
        }
    }
}

fn human_bytes(b:u64)->String{
    let f=b as f64;const KB:f64=1024.0;const MB:f64=KB*1024.0;const GB:f64=MB*1024.0;
    if f>=GB{format!("{:.1} GB",f/GB)}else if f>=MB{format!("{:.1} MB",f/MB)}else if f>=KB{format!("{:.1} KB",f/KB)}else{format!("{b} B")}
}
fn load_directory(dir:PathBuf,tx:Sender<ImgMsg>){
    WalkDir::new(dir).into_iter().filter_map(Result::ok).filter(|e|e.file_type().is_file()).map(|e|e.path().to_owned()).collect::<Vec<_>>().par_iter().for_each(|path|{
        if let Ok(r)=ImageReader::open(path){if let Ok(img)=r.decode(){let _=tx.send((path.clone(),img));}}
    });
}