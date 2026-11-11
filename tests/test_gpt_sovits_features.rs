//! GPT-SoVITS P2（HuBERT + RoBERTa）结构自检。
//!
//! 这里只验证「能加载、形状对、数值正常、和前端能拼起来」。**数值对齐**需要 torch 参考输出：
//!
//! ```bash
//! # 在能跑 GPT-SoVITS 的环境里执行（导出参考特征）
//! python scripts/gpt_sovits/export_reference_features.py --gsv-root D:/GPT-SoVITS --out ref_feats.npz
//! # 再在 Rust 侧比对
//! AHA_GPT_SOVITS_DIR=... AHA_GSV_REF_FEATURES=ref_feats.npz \
//!   cargo test --features gpt-sovits --test test_gpt_sovits_features -- --nocapture
//! ```
//!
//! 没有设置 `AHA_GPT_SOVITS_DIR` 时跳过。

use std::path::PathBuf;

use aha::models::gpt_sovits::{Hubert, RobertaEncoder, WordPieceTokenizer, ZhFrontend};
use anyhow::Result;
use candle_core::{DType, Device, Tensor};

fn model_dir() -> Option<PathBuf> {
    let dir = std::env::var("AHA_GPT_SOVITS_DIR").ok()?;
    let path = PathBuf::from(dir);
    path.join("hubert.safetensors").is_file().then_some(path)
}

fn tensor_stats(name: &str, t: &Tensor) -> Result<()> {
    let dims = t.dims().to_vec();
    let values: Vec<f32> = t.flatten_all()?.to_vec1()?;
    let n = values.len() as f64;
    let mean = values.iter().map(|v| *v as f64).sum::<f64>() / n;
    let var = values
        .iter()
        .map(|v| {
            let d = *v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    let finite = values.iter().all(|v| v.is_finite());
    let max_abs = values.iter().fold(0f32, |m, v| m.max(v.abs()));
    println!(
        "  {name}: dims={dims:?} mean={mean:+.5} std={:.5} max|x|={max_abs:.4} finite={finite}",
        var.sqrt()
    );
    assert!(finite, "{name} 出现了 NaN/Inf");
    Ok(())
}

#[test]
fn hubert_forward_shape_and_stats() -> Result<()> {
    let Some(dir) = model_dir() else {
        println!("跳过：未设置 AHA_GPT_SOVITS_DIR");
        return Ok(());
    };
    let device = Device::Cpu;
    let hubert = Hubert::load(&dir, &device, DType::F32)?;

    // 1 秒 16 kHz 正弦波（440 Hz）
    let sample_rate = 16_000usize;
    let pcm: Vec<f32> = (0..sample_rate)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.3
        })
        .collect();

    let feats = hubert.forward_pcm(&pcm)?;
    tensor_stats("hubert", &feats)?;

    let (frames, hidden) = feats.dims2()?;
    assert_eq!(hidden, 768, "HuBERT 隐层维度不对");
    assert_eq!(
        hubert.downsample(),
        320,
        "卷积总下采样应为 320（16k → 50 Hz）"
    );
    assert_eq!(frames, 49, "1 秒音频应得到 49 帧");
    Ok(())
}

#[test]
fn roberta_forward_shape_and_tokenizer_alignment() -> Result<()> {
    let Some(dir) = model_dir() else {
        println!("跳过：未设置 AHA_GPT_SOVITS_DIR");
        return Ok(());
    };
    let device = Device::Cpu;
    let tokenizer = WordPieceTokenizer::load(dir.join("bert_tokenizer/tokenizer.json"))?;
    let bert = RobertaEncoder::load(&dir, &device, DType::F32)?;

    let text = "今天天气不错，我们出去走走吧。";
    let ids = tokenizer.encode(text);
    // 每个汉字/标点一个 token
    assert_eq!(ids.len(), text.chars().count() + 2, "token 数与字符数不一致");
    assert_eq!(ids[0], 101, "[CLS] id 应为 101");
    assert_eq!(*ids.last().unwrap(), 102, "[SEP] id 应为 102");

    let hidden = bert.forward(&ids)?;
    tensor_stats("bert", &hidden)?;
    assert_eq!(hidden.dims(), &[1, ids.len(), 1024]);
    Ok(())
}

#[test]
fn bert_phone_level_feature_matches_word2ph() -> Result<()> {
    let Some(dir) = model_dir() else {
        println!("跳过：未设置 AHA_GPT_SOVITS_DIR");
        return Ok(());
    };
    let device = Device::Cpu;
    let frontend = ZhFrontend::load(&dir)?;
    let tokenizer = WordPieceTokenizer::load(dir.join("bert_tokenizer/tokenizer.json"))?;
    let bert = RobertaEncoder::load(&dir, &device, DType::F32)?;

    let text = "今天天气不错，我们出去走走吧。";
    let g2p = frontend.g2p(text)?;
    let feats = bert.phone_level_feature(&g2p.norm_text, &tokenizer, &g2p.word2ph)?;
    tensor_stats("bert_phone_level", &feats)?;

    let (channels, phones) = feats.dims2()?;
    assert_eq!(channels, 1024);
    assert_eq!(phones, g2p.phones.len(), "phone 级特征列数应等于 phone 数");
    assert_eq!(phones, g2p.word2ph.iter().sum::<usize>());
    Ok(())
}
