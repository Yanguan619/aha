//! GPT-SoVITS P2/P3 数值对齐：Rust vs torch 参考（`ref_*.bin`）。
//!
//! 参考由 `scripts/gpt_sovits/export_reference_features.py` 在装了 torch 的环境里导出，
//! 再用 `dump_ref_raw.py` 转成裸 f32/u32。目标：cos > 0.9999。
//!
//! `AHA_GPT_SOVITS_DIR=<模型目录> cargo test --features gpt-sovits --test test_gpt_sovits_ref -- --nocapture`

use std::path::{Path, PathBuf};

use aha::models::gpt_sovits::{Hubert, RobertaEncoder};
use anyhow::Result;
use candle_core::{DType, Device, Tensor};

fn model_dir() -> Option<PathBuf> {
    let dir = std::env::var("AHA_GPT_SOVITS_DIR").ok()?;
    let path = PathBuf::from(dir);
    path.join("ref_bert_layer21.bin").is_file().then_some(path)
}

fn read_f32(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn read_u32(path: &Path) -> Result<Vec<u32>> {
    let bytes = std::fs::read(path)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn compare(name: &str, got: &Tensor, reference: &[f32]) -> Result<f64> {
    let got: Vec<f32> = got.flatten_all()?.to_vec1()?;
    assert_eq!(got.len(), reference.len(), "{name} 元素数不一致");

    let dot: f64 = got
        .iter()
        .zip(reference)
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum();
    let na: f64 = got.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = reference.iter().map(|b| (*b as f64).powi(2)).sum::<f64>().sqrt();
    let cos = dot / (na * nb);

    let max_abs = got
        .iter()
        .zip(reference)
        .map(|(a, b)| (*a - *b).abs())
        .fold(0f32, f32::max);
    let mean_abs: f64 = got
        .iter()
        .zip(reference)
        .map(|(a, b)| (*a - *b).abs() as f64)
        .sum::<f64>()
        / got.len() as f64;

    println!("  {name:<12} cos={cos:.7}  max|Δ|={max_abs:.5}  mean|Δ|={mean_abs:.6}");
    Ok(cos)
}

#[test]
fn bert_matches_torch() -> Result<()> {
    let Some(dir) = model_dir() else {
        println!("跳过：缺少 ref_*.bin");
        return Ok(());
    };
    let device = Device::Cpu;
    let ids = read_u32(&dir.join("ref_bert_ids.bin"))?;
    let reference = read_f32(&dir.join("ref_bert_layer21.bin"))?;

    let bert = RobertaEncoder::load(&dir, &device, DType::F32)?;
    let out = bert.forward(&ids)?; // (1, S, 1024)
    println!("\n== BERT layer21 ==");
    let cos = compare("bert", &out, &reference)?;
    assert!(cos > 0.9999, "BERT 对不上（cos={cos}）");
    Ok(())
}

#[test]
fn hubert_matches_torch() -> Result<()> {
    let Some(dir) = model_dir() else {
        println!("跳过：缺少 ref_*.bin");
        return Ok(());
    };
    let device = Device::Cpu;
    let reference = read_f32(&dir.join("ref_hubert.bin"))?;

    // 与 export_reference_features.py 里完全一致的 1 秒 440 Hz 正弦
    let sr = 16_000usize;
    let pcm: Vec<f32> = (0..sr)
        .map(|i| {
            let t = i as f32 / sr as f32;
            (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.3
        })
        .collect();

    let hubert = Hubert::load(&dir, &device, DType::F32)?;
    let out = hubert.forward_pcm(&pcm)?; // (frames, 768)
    println!("\n== HuBERT ==");
    let (frames, hidden) = out.dims2()?;
    println!("  rust dims={frames}x{hidden}  ref={} 帧", reference.len() / hidden);

    let got: Vec<f32> = out.flatten_all()?.to_vec1()?;
    let cosine = |a: &[f32], b: &[f32]| -> f64 {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
        let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum::<f64>().sqrt();
        dot / (na * nb)
    };

    // 1) 时间轴平移搜索
    for shift in [-2i64, -1, 0, 1, 2] {
        let (gs, rs) = if shift >= 0 {
            let s = shift as usize;
            (&got[s * hidden..], &reference[..reference.len() - s * hidden])
        } else {
            let s = (-shift) as usize;
            (&got[..got.len() - s * hidden], &reference[s * hidden..])
        };
        println!("  shift={shift:+} cos={:.7}", cosine(gs, rs));
    }

    // 2) 逐帧 cos（看是不是只有边缘帧不同）
    let mut worst = (f64::MAX, 0usize);
    let mut sum = 0.0f64;
    for i in 0..frames {
        let a = &got[i * hidden..(i + 1) * hidden];
        let b = &reference[i * hidden..(i + 1) * hidden];
        let c = cosine(a, b);
        sum += c;
        if c < worst.0 {
            worst = (c, i);
        }
    }
    println!("  逐帧 cos: mean={:.5} 最差帧 {}={:.5}", sum / frames as f64, worst.1, worst.0);

    let cos = cosine(&got, &reference);
    println!("  整体 cos={cos:.7}");
    assert!(cos > 0.9999, "HuBERT 对不上（cos={cos}）");
    Ok(())
}
