//! GPT-SoVITS P4（SoVITS 前端）自检：参考音频 → prompt codes / 音色向量。
//!
//! 验证形状和取值范围；音质与数值对齐要等 `ref_feats.npz` 和完整合成链路。
//!
//! 注意：仓库自带的 `jiangjiang.wav` 是 48 kHz 立体声、**约 1.48 秒**，
//! 而 GPT-SoVITS 官方建议参考音频 3~10 秒，所以这里只能验证链路，不代表最佳效果。
//!
//! 没有设置 `AHA_GPT_SOVITS_DIR` 时跳过。

use std::path::PathBuf;

use aha::models::gpt_sovits::{Hubert, Sovits, linear_spectrogram};
use aha::utils::audio_utils::{load_audio_use_hound, resample_simple};
use anyhow::Result;
use candle_core::{DType, Device, Tensor};

fn model_dir() -> Option<PathBuf> {
    let dir = std::env::var("AHA_GPT_SOVITS_DIR").ok()?;
    let path = PathBuf::from(dir);
    path.join("sovits.safetensors").is_file().then_some(path)
}

/// 参考音频：仓库自带的约 3 秒中文语音（48 kHz 立体声）
const REF_AUDIO: &str = "assets/audio/jiangjiang.wav";

fn load_ref(device: &Device, sample_rate: usize) -> Result<Tensor> {
    let (wav, sr) = load_audio_use_hound(PathBuf::from(REF_AUDIO), device)?;
    resample_simple(&wav, sr as i64, sample_rate as i64)
}

#[test]
fn prompt_codes_from_reference_audio() -> Result<()> {
    let Some(dir) = model_dir() else {
        println!("跳过：未设置 AHA_GPT_SOVITS_DIR");
        return Ok(());
    };
    let device = Device::Cpu;
    let hubert = Hubert::load(&dir, &device, DType::F32)?;
    let sovits = Sovits::load(&dir, &device, DType::F32)?;

    let wav = load_ref(&device, 16_000)?;
    let mut samples: Vec<f32> = wav.squeeze(0)?.to_vec1()?;
    let n_ref = samples.len();
    // TTS.py 会在参考音频尾部补 0.3 秒静音
    samples.extend(std::iter::repeat_n(0.0f32, 16_000 * 3 / 10));

    let feats = hubert.forward_pcm(&samples)?; // (frames, 768)
    let (frames, hidden) = feats.dims2()?;
    assert_eq!(hidden, 768);

    let codes = sovits.prompt_codes(&feats.unsqueeze(0)?.transpose(1, 2)?.contiguous()?)?;
    println!(
        "参考音频 {n_ref} 采样点(16k) → HuBERT {frames} 帧 → prompt codes {} 个，前 8 个: {:?}",
        codes.len(),
        &codes[..codes.len().min(8)]
    );

    assert_eq!(codes.len(), frames / 2, "ssl_proj 有 stride=2，codes 应为帧数一半");
    assert!(codes.iter().all(|c| (*c as usize) < 1024), "code 必须 < 1024");
    // 1.48 s 音频 + 0.3 s 静音 → 约 88 帧 → 44 个 code
    assert!(codes.len() > 40, "codes 太少，参考音频可能被截断");
    Ok(())
}

#[test]
fn style_embedding_from_spectrogram() -> Result<()> {
    let Some(dir) = model_dir() else {
        println!("跳过：未设置 AHA_GPT_SOVITS_DIR");
        return Ok(());
    };
    let device = Device::Cpu;
    let sovits = Sovits::load(&dir, &device, DType::F32)?;

    // 32 kHz 参考音频 → 线性幅度谱 (1, 1025, T)
    let wav = load_ref(&device, 32_000)?;
    let spec = linear_spectrogram(&wav.squeeze(0)?, 2048, 640, 2048, &device)?;
    let (_, freq, t) = spec.dims3()?;
    assert_eq!(freq, 1025);
    println!("线性谱: {freq} bins × {t} 帧");

    // v2Pro 需要 20480 维说话人向量；这里用占位值，只验证形状/数值有限
    let sv = if sovits.needs_sv() {
        Some(Tensor::zeros((1, 20_480), DType::F32, &device)?)
    } else {
        None
    };
    let ge = sovits.ge(&spec, sv.as_ref())?;
    println!("ge: {:?}", ge.dims());
    assert_eq!(ge.dims(), &[1, 1024, 1]);

    let values: Vec<f32> = ge.flatten_all()?.to_vec1()?;
    assert!(values.iter().all(|v| v.is_finite()), "ge 出现 NaN/Inf");
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    println!("ge 均值 {mean:+.4}");
    Ok(())
}
