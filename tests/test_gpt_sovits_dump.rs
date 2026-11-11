//! 把 SoVITS 全链路的中间量落盘，供 Python 侧（官方实现）逐级对齐。
//!
//! `AHA_GPT_SOVITS_DIR=<dir> cargo test --features gpt-sovits --test test_gpt_sovits_dump -- --nocapture`

use std::io::Write;
use std::path::{Path, PathBuf};

use aha::models::gpt_sovits::{Hubert, Sovits, SynthOptions, ZhFrontend, linear_spectrogram};
use aha::utils::audio_utils::{load_audio_use_hound, resample_simple};
use anyhow::Result;
use candle_core::{DType, Device, Tensor};

const REF_AUDIO: &str = "assets/audio/jiangjiang.wav";
const TEXT: &str = "你好，今天天气不错。";

fn model_dir() -> Option<PathBuf> {
    let dir = std::env::var("AHA_GPT_SOVITS_DIR").ok()?;
    let path = PathBuf::from(dir);
    path.join("sovits.safetensors").is_file().then_some(path)
}

fn dump_f32(path: &Path, t: &Tensor) -> Result<()> {
    let v: Vec<f32> = t.flatten_all()?.to_vec1()?;
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    std::fs::write(path, bytes)?;
    Ok(())
}

fn dump_u32(path: &Path, v: &[u32]) -> Result<()> {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    std::fs::write(path, bytes)?;
    Ok(())
}

fn stats(name: &str, t: &Tensor) -> Result<()> {
    let v: Vec<f32> = t.flatten_all()?.to_vec1()?;
    let n = v.len() as f64;
    let mean = v.iter().map(|a| *a as f64).sum::<f64>() / n;
    let std = (v.iter().map(|a| (*a as f64 - mean).powi(2)).sum::<f64>() / n).sqrt();
    println!("  rust {name:<10} dims={:?} mean={mean:+.4} std={std:.4}", t.dims());
    Ok(())
}

#[test]
fn dump_sovits_stages() -> Result<()> {
    let Some(dir) = model_dir() else {
        println!("跳过：未设置 AHA_GPT_SOVITS_DIR");
        return Ok(());
    };
    let out = dir.join("ref_dump");
    std::fs::create_dir_all(&out)?;
    let device = Device::Cpu;

    let sovits = Sovits::load(&dir, &device, DType::F32)?;
    let hubert = Hubert::load(&dir, &device, DType::F32)?;
    let frontend = ZhFrontend::load(&dir)?;

    // 参考音频：16k（prompt codes）+ 32k（谱）
    let (wav, sr) = load_audio_use_hound(PathBuf::from(REF_AUDIO), &Device::Cpu)?;
    let wav16 = resample_simple(&wav, sr as i64, 16_000)?;
    let mut samples: Vec<f32> = wav16.squeeze(0)?.to_vec1()?;
    samples.extend(std::iter::repeat_n(0.0f32, 16_000 * 3 / 10));
    let hf = hubert.forward_pcm(&samples)?;
    let codes = sovits.prompt_codes(&hf.unsqueeze(0)?.transpose(1, 2)?.contiguous()?)?;

    // 译音色向量
    // sv（eres2net）：16k、不加静音尾（和官方一致）
    let sv = if sovits.needs_sv() {
        let raw16: Vec<f32> = wav16.squeeze(0)?.to_vec1()?;
        Some(sovits.sv(&raw16)?)
    } else {
        None
    };
    if let Some(sv) = &sv {
        dump_f32(&out.join("rust_sv.bin"), sv)?;
        println!("rust sv {:?}", sv.dims());
    }

    let wav32 = resample_simple(&wav, sr as i64, sovits.sampling_rate() as i64)?;
    let mono = wav32.squeeze(0)?;
    dump_f32(&out.join("wav32.bin"), &mono)?;
    let spec = linear_spectrogram(&mono, 2048, sovits.hop_length(), 2048, &Device::Cpu)?;
    dump_f32(&out.join("spec.bin"), &spec)?;
    let sv = if sovits.needs_sv() {
        Some(Tensor::zeros((1, 20_480), DType::F32, &Device::Cpu)?)
    } else {
        None
    };
    let ge = sovits.ge(&spec, sv.as_ref())?;
    dump_f32(&out.join("ge.bin"), &ge)?;

    // 文本 token + BERT 特征（给 Python 侧的 AR 对齐用）
    let g2p = frontend.g2p(TEXT)?;
    let text_ids = frontend.to_ids(&g2p.phones);
    {
        let tokenizer = aha::models::gpt_sovits::WordPieceTokenizer::load(
            dir.join("bert_tokenizer/tokenizer.json"),
        )?;
        let bert = aha::models::gpt_sovits::RobertaEncoder::load(&dir, &device, DType::F32)?;
        let bert_feat = bert.phone_level_feature(&g2p.norm_text, &tokenizer, &g2p.word2ph)?;
        dump_f32(&out.join("bert_feat.bin"), &bert_feat)?;
        println!("bert_feat {:?}", bert_feat.dims());
    }
    dump_u32(&out.join("codes.bin"), &codes)?;
    dump_u32(&out.join("text_ids.bin"), &text_ids)?;

    println!("codes={} text_ids={}", codes.len(), text_ids.len());
    stats("spec", &spec)?;
    stats("ge", &ge)?;

    // enc_p
    let (m, logs) = sovits.probe_enc_p(&codes, &text_ids, &ge, 1.0)?;
    stats("m", &m)?;
    stats("logs", &logs)?;
    dump_f32(&out.join("m.bin"), &m)?;
    dump_f32(&out.join("logs.bin"), &logs)?;

    // flow（noise_scale=0 → z_p = m）
    let (z_p, z) = sovits.probe_flow(&codes, &text_ids, &ge, 1.0, 0.0)?;
    stats("z", &z)?;
    dump_f32(&out.join("z.bin"), &z)?;

    // dec
    let wav_out = sovits.probe_dec(&z, &ge)?;
    stats("wav", &wav_out)?;
    dump_f32(&out.join("wav.bin"), &wav_out)?;

    // 我们的 greedy AR codes（对齐用）
    {
        let t2s = aha::models::gpt_sovits::Text2Semantic::load(&dir, &device, DType::F32)?;
        let tokenizer = aha::models::gpt_sovits::WordPieceTokenizer::load(
            dir.join("bert_tokenizer/tokenizer.json"),
        )?;
        let bert = aha::models::gpt_sovits::RobertaEncoder::load(&dir, &device, DType::F32)?;
        let bert_feat = bert.phone_level_feature(&g2p.norm_text, &tokenizer, &g2p.word2ph)?;
        let opts = aha::models::gpt_sovits::GenOptions {
            greedy: true,
            seed: 42,
            early_stop_num: Some(240),
            ..Default::default()
        };
        let ar_codes = t2s.infer(&text_ids, &bert_feat, &codes, &opts)?;
        dump_u32(&out.join("rust_ar_codes.bin"), &ar_codes)?;
        println!("rust greedy ar codes: {} 个", ar_codes.len());

        // 第一步的 xy 输入和 logits（对齐 AR 用）
        let (xy, logits) = t2s.probe_first_step(&text_ids, &bert_feat, &codes)?;
        dump_f32(&out.join("rust_ar_xy.bin"), &xy)?;
        dump_f32(&out.join("rust_ar_logits.bin"), &logits)?;
        println!("ar xy {:?} logits {:?}", xy.dims(), logits.dims());
    }

    // 同时跑一遍完整 synthesize（含噪声）供试听
    let audio = sovits.synthesize(&codes, &text_ids, &ge, &SynthOptions::default())?;
    aha::utils::audio_utils::save_wav_mono(
        &audio.squeeze(0)?,
        "target/ref_codes.wav",
        sovits.sampling_rate() as u32,
    )?;

    let mut meta = std::fs::File::create(out.join("meta.txt"))?;
    writeln!(meta, "spec_w={} spec_t={}", spec.dim(1)?, spec.dim(2)?)?;
    writeln!(meta, "ge_c={} ge_t={}", ge.dim(1)?, ge.dim(2)?)?;
    writeln!(meta, "m_c={} m_t={}", m.dim(1)?, m.dim(2)?)?;
    writeln!(meta, "wav_n={}", wav_out.dim(2)?)?;
    println!("已写入 {}", out.display());
    Ok(())
}
