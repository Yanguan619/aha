//! 只跑动作头：读 Python dump 的 fused_v/fused_l，验证动作头和 Python 一致。
//!
//! cargo run --release --features sherpa,laya,vla --example turbovla_action -- <模型目录> <v.bin> <l.bin>

use aha::models::turbovla::TurboVlaModel;
use candle_core::{DType, Device, Tensor};

fn read_f32(path: &str) -> anyhow::Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn main() -> anyhow::Result<()> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| r"E:\models\turbovla\turbovla-libero".into());
    let tmp = std::env::var("TEMP").unwrap_or_else(|_| r"C:\Users\Yanguan\AppData\Local\Temp".into());
    let vpath = std::env::args().nth(2).unwrap_or_else(|| format!(r"{tmp}\turbovla_v.bin"));
    let lpath = std::env::args().nth(3).unwrap_or_else(|| format!(r"{tmp}\turbovla_l.bin"));

    let device = Device::Cpu;
    let model = TurboVlaModel::load(&dir, &device, DType::F32)?;
    let v = read_f32(&vpath)?;
    let l = read_f32(&lpath)?;
    let hidden = 256usize;
    println!("v={} l={}", v.len(), l.len());
    let vt = Tensor::from_vec(v, (1, 512, hidden), &device)?;
    let rows_of_text = l.len() / hidden;
    let lt = Tensor::from_vec(l, (1, rows_of_text, hidden), &device)?;
    let cond = Tensor::cat(&[&vt, &lt], 1)?;

    let actions = model.action_from_tokens(cond, &vec![0f32; 8])?;
    let rows = actions.squeeze(0)?.to_vec2::<f32>()?;
    println!("actions {}×{}", rows.len(), rows[0].len());
    for row in rows.iter().take(2) {
        println!("  {row:?}");
    }
    Ok(())
}
