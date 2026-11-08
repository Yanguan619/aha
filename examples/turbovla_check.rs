//! TurboVLA 自检：加载模型、跑一次前向，打印动作 chunk 的形状和首行。
//!
//! cargo run --release --features sherpa,laya,vla --example turbovla_check -- <模型目录>

use aha::models::turbovla::TurboVlaModel;
use candle_core::{DType, Device};

fn main() -> anyhow::Result<()> {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| r"E:\models\turbovla\turbovla-libero".to_string());
    let device = Device::Cpu;
    println!("加载 {dir}");
    let started = std::time::Instant::now();
    let model = TurboVlaModel::load(&dir, &device, DType::F32)?;
    println!(
        "加载完成 {:.1}s（{} 视角，state={}，horizon={}，action={}）",
        started.elapsed().as_secs_f32(),
        model.num_views(),
        model.state_dim(),
        model.horizon(),
        model.action_dim()
    );

    // 一张确定性的 256×256 图（两个视角用同一张，只验证形状/数值通路）
    let mut img = image::RgbImage::new(256, 256);
    for (x, y, px) in img.enumerate_pixels_mut() {
        *px = image::Rgb([(x % 256) as u8, (y % 256) as u8, 128]);
    }
    let mut png = Vec::new();
    image::DynamicImage::ImageRgb8(img).write_to(
        &mut std::io::Cursor::new(&mut png),
        image::ImageFormat::Png,
    )?;

    let state = vec![0f32; model.state_dim()];
    let t0 = std::time::Instant::now();
    let actions = model.predict(&[&png, &png], "put the bowl on the plate", &state)?;
    println!("前向 {:.1}s", t0.elapsed().as_secs_f32());

    let rows = actions.squeeze(0)?.to_vec2::<f32>()?;
    println!("actions: {}×{}", rows.len(), rows[0].len());
    for (i, row) in rows.iter().take(2).enumerate() {
        println!("  [{i}] {:?}", row);
    }
    Ok(())
}
