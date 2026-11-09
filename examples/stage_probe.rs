//! 临时探针：分段对比「融合 Linear」和「回退路径」，定位从哪一段开始不一致。
//!
//! cargo run --release --features vla,cuda --example stage_probe

use aha::models::turbovla::TurboVlaModel;
use candle_core::{DType, Device, Tensor};

fn diff(a: &Tensor, b: &Tensor) -> anyhow::Result<f32> {
    let d = (a.to_dtype(DType::F32)? - b.to_dtype(DType::F32)?)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    Ok(d)
}

fn main() -> anyhow::Result<()> {
    let dev = Device::new_cuda(0)?;
    let model = TurboVlaModel::load(r"E:\models\turbovla\turbovla-libero", &dev, DType::BF16)?;

    let mut img = image::RgbImage::new(256, 256);
    for (x, y, px) in img.enumerate_pixels_mut() {
        *px = image::Rgb([(x % 256) as u8, (y % 256) as u8, 128]);
    }
    let mut png = Vec::new();
    image::DynamicImage::ImageRgb8(img).write_to(
        &mut std::io::Cursor::new(&mut png),
        image::ImageFormat::Png,
    )?;
    let st = vec![0f32; model.state_dim()];
    let imgs: Vec<&[u8]> = vec![&png; model.num_views()];
    let ins = "put the bowl on the plate";

    let a = model.stages(&imgs, ins, &st)?;
    unsafe { std::env::set_var("TURBOVLA_NO_FUSED_LINEAR", "1") };
    let b = model.stages(&imgs, ins, &st)?;

    println!("text     {:.3e}", diff(&a.text, &b.text)?);
    println!("visual   {:.3e}", diff(&a.visual, &b.visual)?);
    println!("fused_v  {:.3e}", diff(&a.fused_v, &b.fused_v)?);
    println!("action   {:.3e}", diff(&a.actions, &b.actions)?);
    Ok(())
}
