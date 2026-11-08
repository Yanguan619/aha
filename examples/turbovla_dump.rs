//! TurboVLA 中间量 dump（对齐 tools/vla/dump_turbovla.py）。
//!
//! cargo run --release --features sherpa,laya,vla --example turbovla_dump -- <模型目录>

use aha::models::turbovla::TurboVlaModel;
use candle_core::{DType, Device, Tensor};

fn stat(name: &str, t: &Tensor) -> anyhow::Result<()> {
    let dims = t.dims().to_vec();
    let v = t.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let n = v.len() as f64;
    let mean = v.iter().map(|&x| x as f64).sum::<f64>() / n;
    let var = v
        .iter()
        .map(|&x| {
            let d = x as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    let head: Vec<String> = v.iter().take(8).map(|x| format!("{x:.8}")).collect();
    println!(
        "{name} {dims:?} mean={mean:.8} std={:.8} first8=[{}]",
        var.sqrt(),
        head.join(", ")
    );
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| r"E:\models\turbovla\turbovla-libero".to_string());
    let device = Device::Cpu;
    let model = TurboVlaModel::load(&dir, &device, DType::F32)?;

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
    let s = model.stages(&[&png, &png], "put the bowl on the plate", &state)?;

    stat("pixel", &s.pixel)?;
    stat("text", &s.text)?;
    stat("visual", &s.visual)?;
    stat("fused_v", &s.fused_v)?;
    stat("fused_l", &s.fused_l)?;
    stat("action", &s.actions)?;
    let tmp = std::env::var("TEMP").unwrap_or_else(|_| r"C:\Users\Yanguan\AppData\Local\Temp".into());
    let dump = |name: &str, t: &Tensor| -> anyhow::Result<()> {
        let v = t.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(format!(r"{tmp}\turbovla_{name}_rust.bin"), bytes)?;
        Ok(())
    };
    dump("v", &s.fused_v)?;
    dump("l", &s.fused_l)?;
    dump("text", &s.text)?;
    dump("visual", &s.visual)?;
    let (v0, l0) = model.debug_fusion_layer(
        0,
        &s.visual,
        &s.text,
        &s.pad_mask,
        &s.self_attention,
    )?;
    dump("v0", &v0)?;
    dump("l0", &l0)?;
    println!(
        "input_ids {:?}",
        s.input_ids.to_dtype(DType::U32)?.to_vec2::<u32>()?[0]
    );
    println!(
        "position_ids {:?}",
        s.position_ids.to_dtype(DType::U32)?.to_vec2::<u32>()?[0]
    );
    println!(
        "key_padding {:?}",
        s.key_padding.to_dtype(DType::U32)?.to_vec2::<u32>()?[0]
    );
    Ok(())
}
