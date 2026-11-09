//! 临时探针：bf16 下模型对输入 1 LSB 扰动的敏感度（判断输出是不是混沌的）。
//!
//! cargo run --release --features vla,cuda --example chaos_probe

use aha::models::turbovla::TurboVlaModel;
use candle_core::{DType, Device};

fn png(seed: u8) -> anyhow::Result<Vec<u8>> {
    let mut img = image::RgbImage::new(256, 256);
    for (x, y, px) in img.enumerate_pixels_mut() {
        *px = image::Rgb([
            ((x as u8).wrapping_add(seed)),
            ((y as u8).wrapping_mul(3).wrapping_add(seed)),
            128u8.wrapping_add(seed),
        ]);
    }
    let mut out = Vec::new();
    image::DynamicImage::ImageRgb8(img).write_to(
        &mut std::io::Cursor::new(&mut out),
        image::ImageFormat::Png,
    )?;
    Ok(out)
}

fn main() -> anyhow::Result<()> {
    for (dt, name) in [(DType::BF16, "bf16"), (DType::F32, "f32")] {
        let dev = Device::new_cuda(0)?;
        let model = TurboVlaModel::load(r"E:\models\turbovla\turbovla-libero", &dev, dt)?;
        let st = vec![0f32; model.state_dim()];
        let ins = "put the bowl on the plate";
        let a = png(0)?;
        let b = png(1)?;
        let ia: Vec<&[u8]> = vec![&a; model.num_views()];
        let ib: Vec<&[u8]> = vec![&b; model.num_views()];
        let ra = model.predict(&ia, ins, &st)?.to_dtype(DType::F32)?;
        let rb = model.predict(&ib, ins, &st)?.to_dtype(DType::F32)?;
        let d = (&ra - &rb)?.abs()?.max_all()?.to_scalar::<f32>()?;
        let scale = ra.abs()?.max_all()?.to_scalar::<f32>()?;
        println!("{name}: 输入差 1 LSB → 动作最大差 {d:.4e}（动作幅度 {scale:.2}）");
    }
    Ok(())
}
