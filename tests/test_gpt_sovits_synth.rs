//! GPT-SoVITS 端到端冒烟测试：文本 + 参考音频 → 32 kHz 波形。
//!
//! ```text
//! 前端 → BERT 特征 → AR(s1) 出语义 codes → SoVITS(enc_p/flow/dec) → wav
//! ```
//!
//! 只验证「能出声、数值正常」，音质要人耳判断；产出的 wav 在
//! `target/gpt_sovits_smoke.wav`，可以自己听。
//!
//! 没有设置 `AHA_GPT_SOVITS_DIR` 时跳过。

use std::path::PathBuf;

use aha::models::gpt_sovits::{
    GenOptions, Hubert, RobertaEncoder, Sovits, SynthOptions, Text2Semantic, WordPieceTokenizer,
    ZhFrontend, linear_spectrogram,
};
use aha::utils::audio_utils::{load_audio_use_hound, save_wav_mono};
use anyhow::Result;
use candle_core::{DType, Device, Tensor};

fn model_dir() -> Option<PathBuf> {
    let dir = std::env::var("AHA_GPT_SOVITS_DIR").ok()?;
    let path = PathBuf::from(dir);
    path.join("sovits.safetensors").is_file().then_some(path)
}

const REF_AUDIO: &str = "assets/audio/jiangjiang.wav";
const TEXT: &str = "你好，今天天气不错。";

#[test]
fn end_to_end_synthesis() -> Result<()> {
    let Some(dir) = model_dir() else {
        println!("跳过：未设置 AHA_GPT_SOVITS_DIR");
        return Ok(());
    };
    let device = Device::Cpu;

    // ---- 1. 前端 ---------------------------------------------------------
    let frontend = ZhFrontend::load(&dir)?;
    let g2p = frontend.g2p(TEXT)?;
    let text_ids = frontend.to_ids(&g2p.phones);
    println!("phones: {} 个，norm_text: {}", text_ids.len(), g2p.norm_text);

    // ---- 2. 参考音频 → prompt codes + ge --------------------------------
    let hubert = Hubert::load(&dir, &device, DType::F32)?;
    let sovits = Sovits::load(&dir, &device, DType::F32)?;

    let (wav, sr) = load_audio_use_hound(PathBuf::from(REF_AUDIO), &device)?;
    let wav16 = aha::utils::audio_utils::resample_simple(&wav, sr as i64, 16_000)?;
    let mut samples: Vec<f32> = wav16.squeeze(0)?.to_vec1()?;
    samples.extend(std::iter::repeat_n(0.0f32, 16_000 * 3 / 10)); // 0.3 s 静音

    let hf = hubert.forward_pcm(&samples)?;
    let prompt = sovits.prompt_codes(&hf.unsqueeze(0)?.transpose(1, 2)?.contiguous()?)?;
    println!("prompt codes: {} 个", prompt.len());

    let wav32 = aha::utils::audio_utils::resample_simple(&wav, sr as i64, sovits.sampling_rate() as i64)?;
    let spec = linear_spectrogram(&wav32.squeeze(0)?, 2048, sovits.hop_length(), 2048, &device)?;
    let sv = if sovits.needs_sv() {
        Some(Tensor::zeros((1, 20_480), DType::F32, &device)?)
    } else {
        None
    };
    let ge = sovits.ge(&spec, sv.as_ref())?;

    // ---- 3. 文本特征 + AR ----------------------------------------------
    let tokenizer = WordPieceTokenizer::load(dir.join("bert_tokenizer/tokenizer.json"))?;
    let bert = RobertaEncoder::load(&dir, &device, DType::F32)?;
    let bert_feat = bert.phone_level_feature(&g2p.norm_text, &tokenizer, &g2p.word2ph)?;

    let t2s = Text2Semantic::load(&dir, &device, DType::F32)?;
    let opts = GenOptions {
        greedy: true,
        max_new_tokens: 40,
        early_stop_num: Some(30),
        ..Default::default()
    };
    let codes = t2s.infer(&text_ids, &bert_feat, &prompt, &opts)?;
    println!("AR 生成语义 token: {} 个", codes.len());
    assert!(!codes.is_empty(), "AR 没有生成任何 token");

    // ---- 4. 合成 ---------------------------------------------------------
    let synth = SynthOptions::default();
    let audio = sovits.synthesize(&codes, &text_ids, &ge, &synth)?;
    let n = audio.dim(2)?;
    let values: Vec<f32> = audio.flatten_all()?.to_vec1()?;
    let peak = values.iter().fold(0f32, |m, v| m.max(v.abs()));
    println!(
        "波形: {} 采样点 @ {} Hz（{:.2} s），峰值 {:.4}",
        n,
        sovits.sampling_rate(),
        n as f32 / sovits.sampling_rate() as f32,
        peak
    );

    assert!(values.iter().all(|v| v.is_finite()), "波形里有 NaN/Inf");
    assert!(n > 1000, "波形太短");
    assert!(peak > 1e-3, "波形几乎静音（峰值 {peak}）");

    std::fs::create_dir_all("target")?;
    save_wav_mono(&audio.squeeze(0)?, "target/gpt_sovits_smoke.wav", sovits.sampling_rate() as u32)?;
    println!("已保存 target/gpt_sovits_smoke.wav");
    Ok(())
}
