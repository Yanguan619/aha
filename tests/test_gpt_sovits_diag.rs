//! GPT-SoVITS 质量诊断：把每一级的数值摊开，判断问题出在「音色」还是「合成数学」。
//!
//! 关注的量：
//!
//! | 量 | 说明 | 正常范围 |
//! |----|------|----------|
//! | prompt codes 唯一率 | HuBERT+量化是否退化 | 唯一值 > 80% |
//! | 量化往返相对误差 | `decode(encode(x))` vs `ssl_proj(x)` | < 0.15 |
//! | `logs` | flow 前的对数方差 | 大约 -4 ~ 0 |
//! | 波形低频能量占比 | 语音应集中在低频 | > 0.5 |
//! | 过零率 / 谱平坦度 | 噪声 vs 语音 | 语音谱平坦度低 |
//!
//! 运行：
//! `AHA_GPT_SOVITS_DIR=... cargo test --features gpt-sovits --test test_gpt_sovits_diag -- --nocapture`

use std::path::PathBuf;

use aha::models::gpt_sovits::{
    GenOptions, Hubert, RobertaEncoder, Sovits, SynthOptions, Text2Semantic, WordPieceTokenizer,
    ZhFrontend, linear_spectrogram,
};
use aha::utils::audio_utils::{load_audio_use_hound, save_wav_mono};
use anyhow::Result;
use candle_core::{DType, Device, Tensor};

/// 编译了 cuda 就用 GPU
fn pick_device() -> Device {
    #[cfg(feature = "cuda")]
    {
        Device::new_cuda(0).unwrap_or(Device::Cpu)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Device::Cpu
    }
}

fn model_dir() -> Option<PathBuf> {
    let dir = std::env::var("AHA_GPT_SOVITS_DIR").ok()?;
    let path = PathBuf::from(dir);
    path.join("sovits.safetensors").is_file().then_some(path)
}

const REF_AUDIO: &str = "assets/audio/jiangjiang.wav";
const TEXT: &str = "你好，今天天气不错。";

fn stats(name: &str, t: &Tensor) -> Result<()> {
    let v: Vec<f32> = t.flatten_all()?.to_vec1()?;
    let n = v.len() as f64;
    let mean = v.iter().map(|x| *x as f64).sum::<f64>() / n;
    let var = v.iter().map(|x| (*x as f64 - mean).powi(2)).sum::<f64>() / n;
    let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let finite = v.iter().all(|x| x.is_finite());
    println!(
        "  {name:<18} dims={:?} mean={mean:+.4} std={:.4} min={min:+.3} max={max:+.3} finite={finite}",
        t.dims(),
        var.sqrt()
    );
    Ok(())
}

#[test]
fn diagnose() -> Result<()> {
    let Some(dir) = model_dir() else {
        println!("跳过：未设置 AHA_GPT_SOVITS_DIR");
        return Ok(());
    };
    let device = pick_device();
    println!("device: {device:?}");

    // ---------------- 参考音频 → HuBERT ----------------
    let hubert = Hubert::load(&dir, &device, DType::F32)?;
    let sovits = Sovits::load(&dir, &device, DType::F32)?;
    // 音频解码/重采样/STFT 都留在 CPU（GPU 上逐帧 FFT 反而慢）
    let (wav, sr) = load_audio_use_hound(PathBuf::from(REF_AUDIO), &Device::Cpu)?;
    let wav16 = aha::utils::audio_utils::resample_simple(&wav, sr as i64, 16_000)?;
    let mut samples: Vec<f32> = wav16.squeeze(0)?.to_vec1()?;
    samples.extend(std::iter::repeat_n(0.0f32, 16_000 * 3 / 10));
    let hf = hubert.forward_pcm(&samples)?;

    println!("\n== 参考音频链路 ==");
    stats("hubert", &hf)?;

    let hin = hf.unsqueeze(0)?.transpose(1, 2)?.contiguous()?;
    let prompt = sovits.prompt_codes(&hin)?;
    let projected = sovits.project(&hin)?;
    let roundtrip = sovits.decode_codes(&prompt)?;

    let mut uniq: Vec<u32> = prompt.clone();
    uniq.sort_unstable();
    uniq.dedup();
    let mut counts = std::collections::HashMap::new();
    for c in &prompt {
        *counts.entry(*c).or_insert(0usize) += 1;
    }
    let top = counts.values().copied().max().unwrap_or(0);
    println!(
        "  prompt codes      {} 个，唯一 {}（{:.1}%），最高频值占比 {:.1}%",
        prompt.len(),
        uniq.len(),
        100.0 * uniq.len() as f32 / prompt.len() as f32,
        100.0 * top as f32 / prompt.len() as f32
    );
    let diff = (&projected - &roundtrip)?;
    let rel = (diff.sqr()?.sum_all()?.to_scalar::<f32>()?
        / projected.sqr()?.sum_all()?.to_scalar::<f32>()?)
    .sqrt();
    println!("  量化往返相对误差   {rel:.4}（< 0.15 属正常）");

    // ---------------- 音色向量 ----------------
    let wav32 = aha::utils::audio_utils::resample_simple(&wav, sr as i64, 32_000)?;
    let spec = linear_spectrogram(&wav32.squeeze(0)?, 2048, 640, 2048, &Device::Cpu)?
        .to_device(&device)?;
    let sv = if sovits.needs_sv() {
        Some(Tensor::zeros((1, 20_480), DType::F32, &device)?)
    } else {
        None
    };
    let ge = sovits.ge(&spec, sv.as_ref())?;

    println!("\n== 音色向量 ==");
    stats("spec(前704)", &spec.narrow(1, 0, 704)?)?;
    stats("ge", &ge)?;

    // ---------------- AR ----------------
    let frontend = ZhFrontend::load(&dir)?;
    let g2p = frontend.g2p(TEXT)?;
    let text_ids = frontend.to_ids(&g2p.phones);
    let tokenizer = WordPieceTokenizer::load(dir.join("bert_tokenizer/tokenizer.json"))?;
    let bert = RobertaEncoder::load(&dir, &device, DType::F32)?;
    let bert_feat = bert.phone_level_feature(&g2p.norm_text, &tokenizer, &g2p.word2ph)?;
    let t2s = Text2Semantic::load(&dir, &device, DType::F32)?;

    println!("\n== AR ==");
    stats("bert_feature", &bert_feat)?;
    for greedy in [true, false] {
        let opts = GenOptions {
            greedy,
            seed: 42,
            early_stop_num: Some(240),
            ..Default::default()
        };
        let codes = t2s.infer(&text_ids, &bert_feat, &prompt, &opts)?;
        println!(
            "  greedy={greedy:<5} → {} codes（{:.2} s 音频），前 12: {:?}",
            codes.len(),
            codes.len() as f32 * 2.0 * 640.0 / 32_000.0,
            &codes[..codes.len().min(12)]
        );
    }

    // ---------------- 合成 + 波形指标 ----------------
    let opts = GenOptions {
        greedy: true,
        seed: 42,
        early_stop_num: Some(240),
        ..Default::default()
    };
    let codes = t2s.infer(&text_ids, &bert_feat, &prompt, &opts)?;
    // 先看 enc_p / flow 的中间量：正常时 logs 应该在 -4 ~ +1 之间
    let (m, logs) = sovits.probe_enc_p(&codes, &text_ids, &ge, 1.0)?;
    stats("m", &m)?;
    stats("logs", &logs)?;
    stats("exp(logs)", &logs.exp()?)?;

    let (z_p, z) = sovits.probe_flow(&codes, &text_ids, &ge, 1.0, 0.0)?;
    stats("z_p(noise=0)", &z_p)?;
    stats("z(flow 输出)", &z)?;

    // 声码器体检：分别用 真实 ge / 零 ge 各跑 zeros & ones，看输出对输入的敏感度
    let ge_zero = Tensor::zeros((1, 1024, 1), DType::F32, &device)?;
    println!("\n-- dec 逐级能量（AHA_GSV_DEBUG=1 时打印）--");
    let z_probe = Tensor::full(1.0f32, (1, 192, 64), &device)?;
    let _ = sovits.probe_dec(&z_probe, &ge)?;
    for (label, g) in [("ge", &ge), ("ge=0", &ge_zero)] {
        for (name, value) in [("zeros", 0.0f32), ("ones", 1.0f32)] {
            let z = Tensor::full(value, (1, 192, 32), &device)?;
            let out = sovits.probe_dec(&z, g)?;
            stats(&format!("dec({name},{label})"), &out)?;
        }
    }

    // 用零 ge 走完整链路：若明显更像人声，说明 ge（缺 eres2net）是噪声主因
    let (z_p0, z0) = sovits.probe_flow(&codes, &text_ids, &ge_zero, 1.0, 0.0)?;
    stats("z_p(ge=0)", &z_p0)?;
    stats("z(ge=0)", &z0)?;
    let out0 = sovits.probe_dec(&z0, &ge_zero)?;
    println!("-- 零 ge 的 dec 输出 --");
    report(&out0, sovits.sampling_rate())?;

    // 关键对照 2：greedy 会退化成 240 个相同 token（等价给声码器一个常量 latent → 嗡声），
    // 所以这里分别用「采样得到的多样 code」和「greedy 的重复 code」各合成一次。
    let sampled_codes = {
        let opts = GenOptions {
            greedy: false,
            seed: 42,
            early_stop_num: Some(240),
            ..Default::default()
        };
        t2s.infer(&text_ids, &bert_feat, &prompt, &opts)?
    };
    {
        let mut uniq = sampled_codes.clone();
        uniq.sort_unstable();
        uniq.dedup();
        println!(
            "\n-- 采样 code: {} 个，唯一 {} 个 --",
            sampled_codes.len(),
            uniq.len()
        );
        let audio = sovits.synthesize(&sampled_codes, &text_ids, &ge, &SynthOptions::default())?;
        report(&audio, sovits.sampling_rate())?;
    }

    // 关键对照：noise_scale=0 时 z_p 就是 m。若这时仍然是噪声，问题在 enc_p/flow/dec，
    // 不在「噪声项被 exp(logs) 放大」。
    for noise in [0.0f32, 0.5] {
        let audio = sovits.synthesize(
            &codes,
            &text_ids,
            &ge,
            &SynthOptions {
                noise_scale: noise,
                ..Default::default()
            },
        )?;
        println!("\n== 合成 noise_scale={noise} ==");
        report(&audio, sovits.sampling_rate())?;
        let flat: Vec<f32> = audio.flatten_all()?.to_vec1()?;
        println!("  前 12 个样本: {:?}", &flat[..12]);
    }
    Ok(())
}

fn report(audio: &Tensor, sr: usize) -> Result<()> {
    stats("waveform", audio)?;
    let flat: Vec<f32> = audio.flatten_all()?.to_vec1()?;
    let n = flat.len();
    let rms = (flat.iter().map(|x| x * x).sum::<f32>() / n as f32).sqrt();
    let peak = flat.iter().fold(0f32, |m, x| m.max(x.abs()));
    let silent = flat.iter().filter(|x| x.abs() < 1e-3).count() as f32 / n as f32;

    // 过零率
    let zcr = flat.windows(2).filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0)).count() as f32 / n as f32;

    // 用 Goertzel 粗估低频(200Hz)/中频(1kHz)/高频(5kHz) 能量比
    let energy_at = |freq: f32| -> f32 {
        let w = 2.0 * std::f32::consts::PI * freq / sr as f32;
        let (mut re, mut im) = (0f32, 0f32);
        for (i, x) in flat.iter().take(16_000).enumerate() {
            re += x * (w * i as f32).cos();
            im -= x * (w * i as f32).sin();
        }
        (re * re + im * im).sqrt() / 1000.0
    };
    let (e200, e1k, e5k) = (energy_at(200.0), energy_at(1000.0), energy_at(5000.0));
    let total = e200 + e1k + e5k + 1e-6;

    println!("  RMS {rms:.4}  峰值 {peak:.4}  静音占比 {:.1}%  过零率 {zcr:.4}", silent * 100.0);
    println!(
        "  能量分布: 200Hz {:.0}%  1kHz {:.0}%  5kHz {:.0}%",
        100.0 * e200 / total,
        100.0 * e1k / total,
        100.0 * e5k / total
    );

    std::fs::create_dir_all("target")?;
    let name = format!("target/diag_{}.wav", (flat.len() as f32 / sr as f32 * 100.0) as u32);
    save_wav_mono(&audio.squeeze(0)?, &name, sr as u32)?;
    println!("  → {name}");
    Ok(())
}
