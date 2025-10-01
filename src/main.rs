#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod gui;
mod sort;
mod load;

use std::{env, path::PathBuf};
use eframe::{egui::ViewportBuilder, NativeOptions};

fn main() -> eframe::Result<()> {
    // Optional CLI arg: a file or folder to open.
    let start_path = env::args().nth(1).map(PathBuf::from);

    // GUI opts: start windowed with decorations. We manage borderless FS ourselves.
    let mut opts = NativeOptions::default();
    opts.viewport = ViewportBuilder::default()
        .with_inner_size([960.0, 540.0])
        .with_decorations(true);

    eframe::run_native(
        "Rust Multicore Image Viewer",
        opts,
        Box::new(move |cc| {
            let mut app = gui::ViewerApp::new(cc.egui_ctx.clone());
            if let Some(p) = &start_path {
                if p.is_file() { app.spawn_loader_file(p.clone()); }
                else if p.is_dir() { app.spawn_loader(p.clone()); }
            }
            Box::new(app)
        }),
    )
}
