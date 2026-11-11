//! 宽松 GGUF 头解析冒烟测试（index-tts2.5 的 audiocpp 打包）。
//!
//! ```text
//! # 只看头：报告 + 组件汇总（截断的文件头也能跑）
//! set AHA_GGUF=D:\models\index-tts2_5-q4_k.gguf
//! cargo test --test test_gguf_lenient -- --nocapture
//!
//! # 额外把每个张量真正读一遍（需要完整文件）
//! cargo test --test test_gguf_lenient -- --ignored --nocapture
//! ```

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::PathBuf;

use aha::models::bigvgan::{BigVGAN, config::BigVGANConfig};
use aha::models::campplus::CAMPPlus;
use aha::models::common::{gguf::Gguf, gguf_lenient, gguf_var_builder::GgufVarBuilder};
use aha::models::index_tts::{
    config::IndexTtsConfig,
    emo::EmoConditioner,
    frontend,
    gpt::IndexTtsGpt,
    pipeline::{GenOptions, IndexTtsPipeline},
    pth_loader,
    s2mel::S2Mel,
    tokenizer::{TextTokenizer, lang_to_token},
};
use aha::models::mask_gct::{config::SemanticCodec, model::RepCodec};
use aha::models::qwen3::{config::Qwen3Config, model::Qwen3Model};
use aha::models::w2v_bert_2_0::{config::W2VBert2_0Config, model::W2VBert2_0Model};
use anyhow::Result;
use candle_core::pickle::read_all_with_key;
use candle_core::{D, DType, Device, Tensor};
use candle_nn::{Module, VarBuilder};

fn model_path() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var("AHA_GGUF").ok()?);
    p.exists().then_some(p)
}

#[test]
fn header_report() -> Result<()> {
    let Some(path) = model_path() else {
        println!("跳过：未设置 AHA_GGUF（指向 index-tts2_5-q4_k.gguf）");
        return Ok(());
    };

    let mut file = File::open(&path)?;
    let lc = gguf_lenient::read(&mut file)?;

    println!("文件: {}", path.display());
    println!("gguf version : {}", lc.report.version);
    println!("tensor_count : {}", lc.report.tensor_count);
    println!("metadata_kv  : {}", lc.report.metadata_count);
    println!("renamed      : {}", lc.report.renamed);
    println!("data_offset  : {}", lc.report.tensor_data_offset);
    println!("skipped      : {:?}", lc.report.skipped_by_dtype());
    for s in lc.report.skipped.iter().take(3) {
        println!("  skip {} {:?} {}", s.name, s.shape, s.dtype);
    }

    println!("\n组件汇总:");
    print!("{}", lc.summary());

    // 内嵌文件（大块不进 Value，走 byte_arrays）
    let files = gguf_lenient::embedded_files(&mut file, &lc)?;
    println!("\n内嵌文件 {} 个:", files.len());
    for (name, bytes) in &files {
        println!("  {name:<40} {:>9} bytes", bytes.len());
    }

    // 关键断言：I64 必须被跳过而不是让整个文件打不开
    assert!(lc.report.skipped.iter().all(|s| s.dtype == "I64"));
    assert!(!lc.content.tensor_infos.is_empty());

    // 原始名还原生效（audiocpp 才有；别的 GGUF 不强制）
    if lc.report.renamed > 0 {
        assert!(
            lc.content
                .tensor_infos
                .keys()
                .any(|k| k.starts_with("bigvgan/") || k.starts_with("gpt/")),
            "应该能拿到原始张量名"
        );
        // index-tts2.5 的 7 个组件都必须能在 map 里找到
        for comp in [
            "bigvgan",
            "wav2vec2bert",
            "gpt",
            "s2mel",
            "semantic_codec",
            "campplus",
            "qwen_emotion",
        ] {
            let n = lc
                .content
                .tensor_infos
                .keys()
                .filter(|k| k.starts_with(&format!("{comp}/")))
                .count();
            assert!(n > 0, "组件 {comp} 没找到张量");
        }
    }

    // 组件 -> dtype 汇总，方便和实现对照
    let mut per: HashMap<String, HashMap<String, usize>> = HashMap::new();
    let mut seen = HashSet::new();
    for (name, info) in lc.content.tensor_infos.iter() {
        if !seen.insert(info.offset) {
            continue;
        }
        let comp = name.split('/').next().unwrap_or(name).to_string();
        *per.entry(comp)
            .or_default()
            .entry(format!("{:?}", info.ggml_dtype))
            .or_insert(0) += 1;
    }
    assert!(!per.is_empty());
    Ok(())
}

/// 用 GGUF 里的权重直接实例化 aha 现有 `CAMPPlus`（说话人向量编码器）
#[test]
#[ignore]
fn campplus_forward() -> Result<()> {
    let Some(path) = model_path() else {
        println!("跳过：未设置 AHA_GGUF");
        return Ok(());
    };
    let mut file = File::open(&path)?;
    let lc = gguf_lenient::read(&mut file)?;
    let device = aha::utils::get_device(None);

    let gguf = Gguf::new(lc.into_content(), file, device.clone());
    let vb = VarBuilder::from_backend(
        Box::new(GgufVarBuilder::new(gguf, "campplus/speaker_encoder")),
        DType::F32,
        device.clone(),
    );
    // funasr CAMPPlus 标准配置：80 维 fbank -> 192 维说话人向量
    let model = CAMPPlus::new(vb, 80, 192, 32, 4, 128)?;

    let feat = Tensor::randn(0f32, 1f32, (1, 300, 80), &device)?;
    let emb = model.forward(&feat)?;
    let norm = emb.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
    println!(
        "campplus: fbank {:?} -> embedding {:?}, L2={norm:.4}",
        feat.dims(),
        emb.dims()
    );
    assert_eq!(emb.dims(), &[1, 192]);
    Ok(())
}

/// 文本分词器：和 Python tiktoken 的参考结果逐 id 对齐
#[test]
#[ignore]
fn index_tts_text_tokenizer() -> Result<()> {
    let Some(path) = model_path() else {
        println!("跳过：未设置 AHA_GGUF");
        return Ok(());
    };
    let mut file = File::open(&path)?;
    let lc = gguf_lenient::read(&mut file)?;
    let bytes =
        gguf_lenient::embedded_file(&mut file, &lc, "multilingual_zh_ja_yue_char_del.tiktoken")?
            .ok_or_else(|| anyhow::anyhow!("GGUF 里没有 .tiktoken 词表"))?;
    let tok = TextTokenizer::from_tiktoken_bytes(&bytes)?;
    println!("tiktoken: {} mergeable ranks", tok.n_vocab);
    assert_eq!(tok.n_vocab, 58836);

    // 参考值由 Python 端用同一份词表 + 上游 pat_str 跑出来
    let cases: Vec<(&str, Vec<u32>)> = vec![
        (
            "<|zh|> 你好世界",
            vec![58839, 220, 48934, 50371, 48721, 53743],
        ),
        (
            "<|zh|> 你好，世界！",
            vec![58839, 220, 48934, 50371, 58827, 48721, 53743, 58824],
        ),
        (
            "<|zh|> 今天天气不错",
            vec![58839, 220, 48838, 50324, 50324, 52692, 48714, 57578],
        ),
        ("<|en|> hello world", vec![58838, 7627, 1002]),
        (
            "<|en|> It's a test, isn't it?",
            vec![58838, 467, 311, 257, 1500, 11, 1939, 380, 309, 30],
        ),
        ("你好世界", vec![48934, 50371, 48721, 53743]),
        ("hello world 123", vec![675, 1909, 1002, 33350]),
    ];
    for (text, expect) in cases {
        let got = tok.encode(text)?;
        println!("{text:?} -> {got:?}");
        assert_eq!(got, expect, "分词不一致: {text:?}");
        assert_eq!(tok.decode(&got)?, text, "解码回环不一致: {text:?}");
    }

    // lang_embedding 下标
    assert_eq!(lang_to_token("zh"), 1);
    assert_eq!(lang_to_token("en"), 0);
    println!(
        "lang_to_token: zh={} en={}",
        lang_to_token("zh"),
        lang_to_token("en")
    );
    Ok(())
}

/// candle 的 ConvTranspose1d 语义自检（PyTorch 是"不翻转核"的转置卷积）
#[test]
fn candle_conv_transpose_sanity() -> Result<()> {
    let d = Device::Cpu;
    // kernel (in=1, out=1, k=3) = [1,2,3]，输入单点 1
    let x = Tensor::from_vec(vec![1f32], (1, 1, 1), &d)?;
    let k = Tensor::from_vec(vec![1f32, 2., 3.], (1, 1, 3), &d)?;
    let y = x.conv_transpose1d(&k, 0, 0, 1, 1, 1)?;
    let v = y.flatten_all()?.to_vec1::<f32>()?;
    println!("conv_transpose1d(delta, [1,2,3]) = {v:?}（PyTorch 期望 [1,2,3]）");
    assert_eq!(v, vec![1., 2., 3.]);
    Ok(())
}

/// CFM 自检：t≈1 时估计器的输出应该接近真实 mel（不含采样循环）
#[test]
#[ignore]
fn index_tts_cfm_sanity() -> Result<()> {
    let Some(path) = model_path() else {
        println!("跳过：未设置 AHA_GGUF");
        return Ok(());
    };
    let ref_wav = std::env::var("AHA_REF_WAV")
        .unwrap_or_else(|_| r"E:\Projects\desktop-pet\tools\tts\sovits\ref\ref.wav".to_string());
    let device = aha::utils::get_device(None);
    let mut pipe = IndexTtsPipeline::from_gguf(&path, &device)?;
    let (audio, sr) = frontend::load_wav(&ref_wav)?;
    let a16 = frontend::resample(&audio, sr, 16000);
    let a22 = frontend::resample(&audio, sr, 22050);
    let feats = frontend::seamless_features(&a16, &device)?;
    let out = pipe.w2v_bert.forward(&feats, None, Some(17), true)?;
    let hidden = out
        .specify_layer_id_hidden_state
        .unwrap_or(out.last_hidden_state);
    let cond_feat = hidden
        .broadcast_sub(&pipe.wav_stats_mean)?
        .broadcast_div(&pipe.wav_stats_std)?;
    let style = pipe
        .campplus
        .forward(&frontend::campplus_fbank(&a16, &device)?)?;
    let x1 = frontend::mel_spectrogram_22050(&a22, &device)?; // (1,80,T)
    let frames = x1.dim(D::Minus1)?;
    let cond = pipe.s2mel.length_regulator.forward(&cond_feat, frames)?;
    let zeros = x1.zeros_like()?;

    for t_val in [0.999f32, 0.7, 0.3] {
        let x_t = (x1.clone() * t_val as f64)?;
        let t = Tensor::from_vec(vec![t_val], (1,), &device)?;
        let pred = pipe
            .s2mel
            .cfm
            .estimator
            .forward(&x_t, &zeros, &t, &style, &cond)?;
        let p = pred.flatten_all()?.to_vec1::<f32>()?;
        let g = x1.flatten_all()?.to_vec1::<f32>()?;
        let n = p.len().min(g.len());
        // 目标速度 u = x1 - (1-σ)z，z=0 → u = x1
        let mae = (0..n).map(|i| (p[i] - g[i]).abs()).sum::<f32>() / n as f32;
        let scale = (0..n).map(|i| g[i].abs()).sum::<f32>() / n as f32;
        println!("t={t_val}: pred vs x1 MAE={mae:.3}（|x1| 均值 {scale:.3}）");
    }

    // 关键：模拟推理时的 prompt 处理（x 的 prompt 段清零），只在非 prompt 区间比较
    {
        let prompt_frames = (0.0837 * 22050.0 / 256.0) as usize; // 占位，下面用真实长度覆盖
        let _ = prompt_frames;
        let pf = x1.dim(D::Minus1)? / 4; // 取 1/4 作为 prompt 区（只为验证）
        let x_t = (x1.clone() * 0.999f64)?;
        let head = x_t.narrow(D::Minus1, 0, pf)?.zeros_like()?;
        let tail = x_t.narrow(D::Minus1, pf, x1.dim(D::Minus1)? - pf)?;
        let x_t = Tensor::cat(&[head, tail], D::Minus1)?;
        let t = Tensor::from_vec(vec![0.999f32], (1,), &device)?;
        let pred = pipe
            .s2mel
            .cfm
            .estimator
            .forward(&x_t, &x1, &t, &style, &cond)?;
        let p = pred.narrow(D::Minus1, pf, x1.dim(D::Minus1)? - pf)?;
        let g = x1.narrow(D::Minus1, pf, x1.dim(D::Minus1)? - pf)?;
        let p = p.flatten_all()?.to_vec1::<f32>()?;
        let g = g.flatten_all()?.to_vec1::<f32>()?;
        let n = p.len().min(g.len());
        let mae = (0..n).map(|i| (p[i] - g[i]).abs()).sum::<f32>() / n as f32;
        println!("带 prompt（x 的 prompt 段清零，只算非 prompt 区）: MAE={mae:.3}");
    }
    Ok(())
}

/// 原始权重精度对比：用官方 BF16 转的 safetensors 替换 Q4_K，看杂音差多少
#[test]
#[ignore]
fn index_tts_orig_precision() -> Result<()> {
    let Some(path) = model_path() else {
        println!("跳过：未设置 AHA_GGUF");
        return Ok(());
    };
    let ref_wav = std::env::var("AHA_REF_WAV")
        .unwrap_or_else(|_| r"E:\Projects\desktop-pet\tools\tts\sovits\ref\ref.wav".to_string());
    let device = aha::utils::get_device(None);
    let mut pipe = IndexTtsPipeline::from_gguf(&path, &device)?;
    let tok = pipe.text_tokenizer(&path)?;

    // 可选：替换 s2mel（CFM+DiT）为原始权重
    let s2mel_st = r"E:\tmp\indextts25-orig\s2mel.safetensors";
    if std::path::Path::new(s2mel_st).exists() {
        let s2mel_dtype = match std::env::var("AHA_S2MEL_DTYPE").ok().as_deref() {
            Some("f16") => DType::F16,
            _ => DType::F32,
        };
        pipe.s2mel = pth_loader::load_s2mel_from_safetensors_dtype(s2mel_st, &device, s2mel_dtype)?;
        pipe.cfm_dtype = s2mel_dtype; // 同步精度标记
        println!("s2mel 已替换为原始权重 dtype={s2mel_dtype:?}");
    }

    // 可选：替换 GPT 为原始权重
    let gpt_st = r"E:\tmp\indextts25-orig\gpt.safetensors";
    if std::path::Path::new(gpt_st).exists() && std::env::var("AHA_ORIG_GPT").is_ok() {
        pipe.gpt = pth_loader::load_gpt_from_safetensors(gpt_st, &device)?;
        println!("GPT 已替换为原始 F32 权重");
    }

    let text =
        std::env::var("AHA_TEXT").unwrap_or_else(|_| "今天天气不错，我们出去走走吧。".to_string());
    let opt = match std::env::var("AHA_MODE").ok().as_deref() {
        Some("fast") => GenOptions::fast(),
        _ => GenOptions {
            max_new_tokens: 200,
            repetition_penalty: std::env::var("AHA_REP")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1.0),
            ..Default::default()
        },
    };

    let steps: usize = std::env::var("AHA_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(25);
    let t0 = std::time::Instant::now();
    let (wav, timings) = pipe.synthesize_profiled(&tok, &ref_wav, &text, "zh", &opt, steps, 0.7)?;
    for (name, ms) in &timings {
        println!("  {name:<22} {ms:>5} ms");
    }
    let secs = wav.len() as f32 / 22050.0;
    println!(
        "合成 {:.2}s 耗时 {:?}（含 {} 步 CFM）",
        secs,
        t0.elapsed(),
        steps
    );

    let out = std::env::temp_dir().join("index_tts_orig_s2mel.wav");
    frontend::write_wav(out.to_str().unwrap(), &wav, 22050)?;
    println!("wrote {}", out.display());

    // 音色对比
    let a16 = frontend::resample(&wav, 22050, 16000);
    let spk = pipe
        .campplus
        .forward(&frontend::campplus_fbank(&a16, &device)?)?;
    let (audio_ref, sr_ref) = frontend::load_wav(&ref_wav)?;
    let b = pipe
        .campplus
        .forward(&frontend::campplus_fbank(
            &frontend::resample(&audio_ref, sr_ref, 16000),
            &device,
        )?)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let a = spk.flatten_all()?.to_vec1::<f32>()?;
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("campplus 余弦 = {:.4}", dot / (na * nb).max(1e-9));
    Ok(())
}

/// 用原始 pth 权重替换 Q4_K 权重做精度对比（s2mel 部分）
#[test]
#[ignore]
fn index_tts_orig_s2mel_keys() -> Result<()> {
    let pth = r"E:\tmp\indextts25-orig\s2mel.safetensors";
    let weights = candle_core::safetensors::load(pth, &Device::Cpu)?;
    println!("s2mel.safetensors: {} 个张量", weights.len());
    let mut keys: Vec<&str> = weights.keys().map(|k| k.as_str()).collect();
    keys.sort();
    for k in keys.iter().take(10) {
        let t = &weights[*k];
        println!("  {k:70} {:?}", t.shape());
    }
    let cfm: Vec<&&str> = keys.iter().filter(|k| k.contains("cfm")).collect();
    println!("\ncfm 相关 {} 个:", cfm.len());
    for k in cfm.iter().take(10) {
        let t = &weights[**k];
        println!("  {k:70} {:?}", t.shape());
    }
    let lr: Vec<&&str> = keys
        .iter()
        .filter(|k| k.contains("length_regulator"))
        .collect();
    println!("\nlength_regulator 相关 {} 个:", lr.len());
    for k in lr.iter().take(10) {
        let t = &weights[**k];
        println!("  {k:70} {:?}", t.shape());
    }
    Ok(())
}

/// 跳过 codec（直接 w2v 特征 → length_regulator → CFM → bigvgan），判断杂音来自哪一段
#[test]
#[ignore]
fn index_tts_bypass_codec() -> Result<()> {
    let Some(path) = model_path() else {
        println!("跳过：未设置 AHA_GGUF");
        return Ok(());
    };
    let ref_wav = std::env::var("AHA_REF_WAV")
        .unwrap_or_else(|_| r"E:\Projects\desktop-pet\tools\tts\sovits\ref\ref.wav".to_string());
    let steps: usize = std::env::var("AHA_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(25);
    let cfg: f64 = std::env::var("AHA_CFG")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.7);
    let device = aha::utils::get_device(None);
    let mut pipe = IndexTtsPipeline::from_gguf(&path, &device)?;

    let (audio, sr) = frontend::load_wav(&ref_wav)?;
    let a16 = frontend::resample(&audio, sr, 16000);
    let a22 = frontend::resample(&audio, sr, 22050);
    let feats = frontend::seamless_features(&a16, &device)?;
    let out = pipe.w2v_bert.forward(&feats, None, Some(17), true)?;
    let hidden = out
        .specify_layer_id_hidden_state
        .unwrap_or(out.last_hidden_state);
    let cond_feat = hidden
        .broadcast_sub(&pipe.wav_stats_mean)?
        .broadcast_div(&pipe.wav_stats_std)?;
    let style = pipe
        .campplus
        .forward(&frontend::campplus_fbank(&a16, &device)?)?;
    let prompt_mel = frontend::mel_spectrogram_22050(&a22, &device)?;
    let prompt_frames = prompt_mel.dim(D::Minus1)?;

    let target = (cond_feat.dim(1)? as f64 * 1.72) as usize;
    let prompt_cond = pipe
        .s2mel
        .length_regulator
        .forward(&cond_feat, prompt_frames)?;
    let cond = pipe.s2mel.length_regulator.forward(&cond_feat, target)?;
    let cat_condition = Tensor::cat(&[&prompt_cond, &cond], 1)?;
    println!(
        "cond_feat {:?} -> target={target}, cat_condition {:?}",
        cond_feat.dims(),
        cat_condition.dims()
    );
    let mel = pipe
        .s2mel
        .cfm
        .solve_euler(&cat_condition, &prompt_mel, &style, steps, cfg, 1.0)?;
    let mel = mel.narrow(D::Minus1, prompt_frames, target)?;
    let wav = pipe.bigvgan.forward(&mel)?;
    let samples = wav.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?;
    let pm = prompt_mel.flatten_all()?.to_vec1::<f32>()?;
    let gen_mel = frontend::mel_spectrogram_22050(&samples, &device)?;
    let gm = gen_mel.flatten_all()?.to_vec1::<f32>()?;
    let n = pm.len().min(gm.len());
    let mae = (0..n).map(|i| (pm[i] - gm[i]).abs()).sum::<f32>() / n as f32;
    println!(
        "steps={steps} cfg={cfg}: {:.2}s, 峰值 {:.3}, mel MAE(对原文)={mae:.3}",
        samples.len() as f32 / 22050.0,
        samples.iter().fold(0f32, |a, v| a.max(v.abs()))
    );
    let out_path = std::env::temp_dir().join(format!("index_tts_bypass_s{steps}_cfg{cfg}.wav"));
    frontend::write_wav(out_path.to_str().unwrap(), &samples, 22050)?;
    println!("wrote {}", out_path.display());
    Ok(())
}

/// 条件消融：说话人 latent 置零后，GPT 生成的 code 是否变化
#[test]
#[ignore]
fn index_tts_cond_ablation() -> Result<()> {
    let Some(path) = model_path() else {
        println!("跳过：未设置 AHA_GGUF");
        return Ok(());
    };
    let ref_wav = std::env::var("AHA_REF_WAV")
        .unwrap_or_else(|_| r"E:\Projects\desktop-pet\tools\tts\sovits\ref\ref.wav".to_string());
    let device = aha::utils::get_device(None);
    let mut pipe = IndexTtsPipeline::from_gguf(&path, &device)?;
    let tok = pipe.text_tokenizer(&path)?;

    let (audio, sr) = frontend::load_wav(&ref_wav)?;
    let a16 = frontend::resample(&audio, sr, 16000);
    let feats = frontend::seamless_features(&a16, &device)?;
    let out = pipe.w2v_bert.forward(&feats, None, Some(17), true)?;
    let hidden = out
        .specify_layer_id_hidden_state
        .unwrap_or(out.last_hidden_state);
    let cond_feat = hidden
        .broadcast_sub(&pipe.wav_stats_mean)?
        .broadcast_div(&pipe.wav_stats_std)?;
    let style = pipe
        .campplus
        .forward(&frontend::campplus_fbank(&a16, &device)?)?;
    println!("style: {:?}", style.dims());
    let s = style.flatten_all()?.to_vec1::<f32>()?;
    println!(
        "style mean={:.4} std={:.4}（若接近 0 说明 campplus 输出异常）",
        s.iter().sum::<f32>() / s.len() as f32,
        (s.iter().map(|v| v * v).sum::<f32>() / s.len() as f32).sqrt()
    );

    let (ids, lang) = pipe.encode_text(&tok, "zh", "你好，世界。")?;
    let opt = GenOptions {
        max_new_tokens: 80,
        ..Default::default()
    };

    let conds = pipe.cond_latents(&cond_feat, &cond_feat, &style, 1.0)?;
    let c = conds.flatten_all()?.to_vec1::<f32>()?;
    println!(
        "conds mean={:.4} std={:.4}",
        c.iter().sum::<f32>() / c.len() as f32,
        (c.iter().map(|v| v * v).sum::<f32>() / c.len() as f32).sqrt()
    );
    let a = pipe.generate(&ids, lang, &conds, &opt)?;
    pipe.gpt.clear_kv_cache();
    let zeros = Tensor::zeros_like(&conds)?;
    let b = pipe.generate(&ids, lang, &zeros, &opt)?;
    let a_v = a.flatten_all()?.to_vec1::<u32>()?;
    let b_v = b.flatten_all()?.to_vec1::<u32>()?;
    let n = a_v.len().min(b_v.len());
    let same = (0..n).filter(|i| a_v[*i] == b_v[*i]).count();
    println!("codes(有条件下) 前 12: {:?}", &a_v[..a_v.len().min(12)]);
    println!("codes(零条件下) 前 12: {:?}", &b_v[..b_v.len().min(12)]);
    println!(
        "长度 {} vs {}，前 {} 个 code 相同率 {:.1}%",
        a_v.len(),
        b_v.len(),
        n,
        same as f32 / n as f32 * 100.0
    );
    // 参考音频的码 vs GPT 生成的码：看"游程长度"（连续相同码）和码表使用情况
    let (ref_codes, _) = pipe.codec.quantize(&cond_feat)?;
    let r_v = ref_codes.flatten_all()?.to_vec1::<u32>()?;
    let stats = |v: &[u32]| {
        let mut runs = 1usize;
        for i in 1..v.len() {
            if v[i] != v[i - 1] {
                runs += 1;
            }
        }
        let mut set = std::collections::HashSet::new();
        for c in v {
            set.insert(*c);
        }
        (set.len(), v.len() as f32 / runs as f32)
    };
    let (rd, rr) = stats(&r_v);
    let (gd, gr) = stats(&a_v);
    println!("参考码: 去重 {rd}，平均游程 {rr:.2}；GPT 码: 去重 {gd}，平均游程 {gr:.2}");
    Ok(())
}

/// 用参考音频自己的语义码重建：验证 codec + length_regulator + CFM + 声码器这条下游链路
#[test]
#[ignore]
fn index_tts_reconstruct_from_ref_codes() -> Result<()> {
    let Some(path) = model_path() else {
        println!("跳过：未设置 AHA_GGUF");
        return Ok(());
    };
    let ref_wav = std::env::var("AHA_REF_WAV")
        .unwrap_or_else(|_| r"E:\Projects\desktop-pet\tools\tts\sovits\ref\ref.wav".to_string());
    let device = aha::utils::get_device(None);
    let mut pipe = IndexTtsPipeline::from_gguf(&path, &device)?;

    let (audio, sr) = frontend::load_wav(&ref_wav)?;
    let a16 = frontend::resample(&audio, sr, 16000);
    let a22 = frontend::resample(&audio, sr, 22050);
    let feats = frontend::seamless_features(&a16, &device)?;
    let out = pipe.w2v_bert.forward(&feats, None, Some(17), true)?;
    let hidden = out
        .specify_layer_id_hidden_state
        .unwrap_or(out.last_hidden_state);
    let cond_feat = hidden
        .broadcast_sub(&pipe.wav_stats_mean)?
        .broadcast_div(&pipe.wav_stats_std)?;
    println!("cond_feat {:?}", cond_feat.dims());

    // 语义码往返
    let (codes, latents) = pipe.codec.quantize(&cond_feat)?;
    println!("codes {:?}, latents {:?}", codes.dims(), latents.dims());
    let rec = pipe.codec.decode_codes(&codes)?;
    // 消融：用连续（未量化）重建替换 codec 解码，判断杂音是否来自码本量化
    let cont = std::env::var("AHA_CODEC_CONT").is_ok();
    let rec = if cont {
        let (r, _) = pipe.codec.forward(&cond_feat)?;
        println!("（消融）使用连续重建，跳过码本量化");
        r
    } else {
        rec
    };
    let n = rec.dim(1)?.min(cond_feat.dim(1)?);
    let a = cond_feat.narrow(1, 0, n)?.flatten_all()?.to_vec1::<f32>()?;
    let b = rec.narrow(1, 0, n)?.flatten_all()?.to_vec1::<f32>()?;
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("codec 往返 cos={:.4}", dot / (na * nb).max(1e-9));
    {
        let (r_cont, _) = pipe.codec.forward(&cond_feat)?;
        let bb = r_cont.narrow(1, 0, n)?.flatten_all()?.to_vec1::<f32>()?;
        let d2: f32 = a.iter().zip(bb.iter()).map(|(x, y)| x * y).sum();
        let nb2: f32 = bb.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!("codec 连续重建 cos={:.4}", d2 / (na * nb2).max(1e-9));
    }

    // 下游：prompt 条件 + CFM + 声码器
    let prompt_frames = a22.len() / 256;
    let prompt_mel = frontend::mel_spectrogram_22050(&a22, &device)?;
    let prompt_cond = pipe
        .s2mel
        .length_regulator
        .forward(&cond_feat, prompt_frames)?;
    let style = pipe
        .campplus
        .forward(&frontend::campplus_fbank(&a16, &device)?)?;
    let steps: usize = std::env::var("AHA_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(25);
    let cfg: f64 = std::env::var("AHA_CFG")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.7);
    let wav = if cont {
        let frames = rec.dim(1)?;
        let target = (frames as f64 * 1.72) as usize;
        let cond = pipe.s2mel.length_regulator.forward(&rec, target)?;
        let cat = Tensor::cat(&[&prompt_cond, &cond], 1)?;
        let mel = pipe
            .s2mel
            .cfm
            .solve_euler(&cat, &prompt_mel, &style, steps, cfg, 1.0)?;
        let mel = mel.narrow(D::Minus1, prompt_mel.dim(D::Minus1)?, target)?;
        pipe.bigvgan.forward(&mel)?
    } else {
        pipe.codes_to_wav(&codes, &prompt_mel, &prompt_cond, &style, steps, cfg)?
    };
    let samples = wav.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?;
    println!(
        "重建 {} 采样点 / {:.2}s，峰值 {:.3}",
        samples.len(),
        samples.len() as f32 / 22050.0,
        samples.iter().fold(0f32, |a, v| a.max(v.abs()))
    );
    let out_path = std::env::temp_dir().join("index_tts_ref_codes.wav");
    frontend::write_wav(out_path.to_str().unwrap(), &samples, 22050)?;
    println!("wrote {}", out_path.display());

    // 客观指标：mel 相似度 + 说话人相似度（拿重建音频的 campplus 和原始音频比）
    let rec_mel = frontend::mel_spectrogram_22050(&samples, &device)?;
    let pm = prompt_mel.flatten_all()?.to_vec1::<f32>()?;
    let rm = rec_mel.flatten_all()?.to_vec1::<f32>()?;
    let n = pm.len().min(rm.len());
    let mae = (0..n).map(|i| (pm[i] - rm[i]).abs()).sum::<f32>() / n as f32;
    let rec_fbank = frontend::campplus_fbank(&frontend::resample(&samples, 22050, 16000), &device)?;
    let rec_spk = pipe.campplus.forward(&rec_fbank)?;
    let s1 = style.flatten_all()?.to_vec1::<f32>()?;
    let s2 = rec_spk.flatten_all()?.to_vec1::<f32>()?;
    let dot: f32 = s1.iter().zip(s2.iter()).map(|(a, b)| a * b).sum();
    let n1: f32 = s1.iter().map(|a| a * a).sum::<f32>().sqrt();
    let n2: f32 = s2.iter().map(|a| a * a).sum::<f32>().sqrt();
    println!(
        "重建 vs 原始：mel MAE={mae:.3}，campplus 余弦={:.4}",
        dot / (n1 * n2).max(1e-9)
    );
    Ok(())
}

/// 声码器自检：用参考音频自己的 mel 重建，和原始音频做相关性对比
#[test]
#[ignore]
fn index_tts_vocoder_check() -> Result<()> {
    let Some(path) = model_path() else {
        println!("跳过：未设置 AHA_GGUF");
        return Ok(());
    };
    let ref_wav = std::env::var("AHA_REF_WAV")
        .unwrap_or_else(|_| r"E:\Projects\desktop-pet\tools\tts\sovits\ref\ref.wav".to_string());
    let device = aha::utils::get_device(None);
    let mut pipe = IndexTtsPipeline::from_gguf(&path, &device)?;
    let (audio, sr) = frontend::load_wav(&ref_wav)?;
    let a22 = frontend::resample(&audio, sr, 22050);
    let mel = frontend::mel_spectrogram_22050(&a22, &device)?;
    let wav = pipe.bigvgan.forward(&mel)?;
    let rec = wav.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?;
    let n = rec.len().min(a22.len());
    let (mut sa, mut sb, mut sab, mut saa, mut sbb) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for i in 0..n {
        let x = a22[i] as f64;
        let y = rec[i] as f64;
        sa += x;
        sb += y;
        sab += x * y;
        saa += x * x;
        sbb += y * y;
    }
    let (ma, mb) = (sa / n as f64, sb / n as f64);
    let cov = sab / n as f64 - ma * mb;
    let va = saa / n as f64 - ma * ma;
    let vb = sbb / n as f64 - mb * mb;
    let corr = cov / (va.sqrt() * vb.sqrt()).max(1e-12);
    println!(
        "重建相关性 corr={corr:.4}，原始 RMS={:.4}，重建 RMS={:.5}",
        va.sqrt(),
        vb.sqrt()
    );
    println!(
        "重建峰值 {:.5}，原始峰值 {:.3}",
        rec.iter().fold(0f32, |a, v| a.max(v.abs())),
        a22.iter().fold(0f32, |a, v| a.max(v.abs()))
    );
    // 用重建音频再算一遍 mel，和输入 mel 比（比波形相关性更能说明声码器是否正常）
    let rec_mel = frontend::mel_spectrogram_22050(&rec, &device)?;
    let a = mel.flatten_all()?.to_vec1::<f32>()?;
    let b = rec_mel.flatten_all()?.to_vec1::<f32>()?;
    let n = a.len().min(b.len());
    let mae = (0..n).map(|i| (a[i] - b[i]).abs()).sum::<f32>() / n as f32;
    println!("重建 mel 与输入 mel 的平均绝对误差 {mae:.3}（单位是 log 幅度）");
    let out = std::env::temp_dir().join("index_tts_vocoder_only.wav");
    frontend::write_wav(out.to_str().unwrap(), &rec, 22050)?;
    println!("wrote {}", out.display());
    Ok(())
}

/// 前端特征数值检查（不发散时用来看特征尺度）
#[test]
#[ignore]
fn index_tts_frontend_stats() -> Result<()> {
    let ref_wav = std::env::var("AHA_REF_WAV")
        .unwrap_or_else(|_| r"E:\Projects\desktop-pet\tools\tts\sovits\ref\ref.wav".to_string());
    let device = aha::utils::get_device(None);
    let (audio, sr) = frontend::load_wav(&ref_wav)?;
    println!(
        "wav: {} 采样 @{}Hz, 峰值 {:.3}",
        audio.len(),
        sr,
        audio.iter().fold(0f32, |a, v| a.max(v.abs()))
    );
    let a16 = frontend::resample(&audio, sr, 16000);
    let a22 = frontend::resample(&audio, sr, 22050);
    println!(
        "resample: 16k {} 采样 峰值 {:.3}, 22.05k {} 采样 峰值 {:.3}",
        a16.len(),
        a16.iter().fold(0f32, |a, v| a.max(v.abs())),
        a22.len(),
        a22.iter().fold(0f32, |a, v| a.max(v.abs()))
    );
    let fbank = frontend::campplus_fbank(&a16, &device)?;
    let feats = frontend::seamless_features(&a16, &device)?;
    let mel = frontend::mel_spectrogram_22050(&a22, &device)?;
    // conv_post 的 weight_norm 数值检查
    {
        let mut f = File::open(&std::env::var("AHA_GGUF")?)?;
        let lc = gguf_lenient::read(&mut f)?;
        let mut g = Gguf::new(lc.into_content(), f, device.clone());
        let wg = g.get_dequantized("bigvgan/conv_post.weight_g")?;
        let wv = g.get_dequantized("bigvgan/conv_post.weight_v")?;
        println!(
            "conv_post weight_g={:?} {:?}, weight_v {:?}",
            wg.dims(),
            wg.flatten_all()?.to_vec1::<f32>()?,
            wv.dims()
        );
        let v = wv.flatten_all()?.to_vec1::<f32>()?;
        println!(
            "weight_v: n={} norm={:.4} mean={:.5} peak={:.5}",
            v.len(),
            v.iter().map(|x| x * x).sum::<f32>().sqrt(),
            v.iter().sum::<f32>() / v.len() as f32,
            v.iter().fold(0f32, |a, x| a.max(x.abs()))
        );
        for name in [
            "bigvgan/activation_post.act.alpha",
            "bigvgan/activation_post.act.beta",
            "bigvgan/resblocks.0.activations.0.act.alpha",
            "bigvgan/resblocks.0.activations.0.act.beta",
            "bigvgan/conv_pre.weight_g",
            "bigvgan/ups.0.0.weight_g",
        ] {
            let t = g.get_dequantized(name)?;
            let v = t.flatten_all()?.to_vec1::<f32>()?;
            println!(
                "{name}: n={} min={:.4} mean={:.4} max={:.4}",
                v.len(),
                v.iter().cloned().fold(f32::INFINITY, f32::min),
                v.iter().sum::<f32>() / v.len() as f32,
                v.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            );
        }
    }
    for (name, t) in [
        ("campplus fbank", &fbank),
        ("seamless", &feats),
        ("ref_mel", &mel),
    ] {
        let v = t.flatten_all()?.to_vec1::<f32>()?;
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!(
            "{name}: {:?} mean={mean:.3} min={min:.3} max={max:.3}",
            t.dims()
        );
    }
    Ok(())
}

/// 真实参考音频 + 真实文本 -> 写 wav（端到端，最慢）
#[test]
#[ignore]
fn index_tts_real_synthesis() -> Result<()> {
    let Some(path) = model_path() else {
        println!("跳过：未设置 AHA_GGUF");
        return Ok(());
    };
    let ref_wav = std::env::var("AHA_REF_WAV")
        .unwrap_or_else(|_| r"E:\Projects\desktop-pet\tools\tts\sovits\ref\ref.wav".to_string());
    let device = aha::utils::get_device(None);
    let t0 = std::time::Instant::now();
    // Q4_K GGUF GPT (F32) + 原始 F32 s2mel + F16 BigVGAN（vocoder 单次前向，F16 通常无损）
    let bv_dtype = match std::env::var("AHA_BIGVGAN_DTYPE").ok().as_deref() {
        Some("f32") => DType::F32,
        _ => DType::F16,
    };
    println!("device={device:?} GPT=F32(GGUF) s2mel=F32(orig) BigVGAN={bv_dtype:?}");
    let mut pipe =
        IndexTtsPipeline::from_gguf_full_split(&path, &device, DType::F32, DType::F32, bv_dtype)?;
    let tok = pipe.text_tokenizer(&path)?;
    println!("加载完成 {:?}", t0.elapsed());

    let opt = GenOptions {
        max_new_tokens: 200,
        do_sample: std::env::var("AHA_GREEDY").is_err(),
        repetition_penalty: std::env::var("AHA_REP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10.0),
        ..Default::default()
    };
    // 消融：把说话人条件置零，看生成的 code 是否变化（验证条件链路是否真的起作用）
    let zero_cond = std::env::var("AHA_ZERO_COND").is_ok();
    let t1 = std::time::Instant::now();
    let text = std::env::var("AHA_TEXT").unwrap_or_else(|_| "你好，世界。".to_string());
    println!("合成文本: {text}");
    let (wav, timings) = pipe.synthesize_profiled(&tok, &ref_wav, &text, "zh", &opt, 25, 0.7)?;
    for (name, ms) in &timings {
        println!("  {name:<22} {ms:>5} ms");
    }
    let secs = wav.len() as f32 / 22050.0;
    println!(
        "合成: {} 采样点 / {:.2}s，耗时 {:?}",
        wav.len(),
        secs,
        t1.elapsed()
    );

    let out = std::env::temp_dir().join("index_tts_out.wav");
    aha::models::index_tts::frontend::write_wav(out.to_str().unwrap(), &wav, 22050)?;
    // 归一化版本：先确认"音量小"是纯增益问题还是内容问题
    let peak = wav.iter().fold(0f32, |a, v| a.max(v.abs()));
    if peak > 1e-6 {
        let gain = 0.9 / peak;
        let boosted: Vec<f32> = wav.iter().map(|v| (v * gain).clamp(-1.0, 1.0)).collect();
        let out_n = std::env::temp_dir().join("index_tts_out_boosted.wav");
        aha::models::index_tts::frontend::write_wav(out_n.to_str().unwrap(), &boosted, 22050)?;
        println!("增益 {gain:.1}x 版本: {}", out_n.display());
    }
    println!("wrote {}", out.display());
    assert!(wav.len() > 1000);
    Ok(())
}

/// 端到端冒烟：条件 -> GPT 自回归 -> codec -> s2mel -> bigvgan（输入用随机数据）
#[test]
#[ignore]
fn index_tts_pipeline_smoke() -> Result<()> {
    let Some(path) = model_path() else {
        println!("跳过：未设置 AHA_GGUF");
        return Ok(());
    };
    let device = aha::utils::get_device(None);
    let mut pipe = IndexTtsPipeline::from_gguf(&path, &device)?;
    println!("pipeline 加载完成，采样率 {}", pipe.sample_rate);

    let spk_w2v = Tensor::randn(0f32, 1f32, (1, 40, 1024), &device)?;
    let emo_w2v = Tensor::randn(0f32, 1f32, (1, 40, 1024), &device)?;
    let campplus = Tensor::randn(0f32, 1f32, (1, 192), &device)?;
    let conds = pipe.cond_latents(&spk_w2v, &emo_w2v, &campplus, 1.0)?;
    println!("conds: {:?}", conds.dims());
    assert_eq!(conds.dims(), &[1, 3, 1280]);

    let prompt_frames = 20;
    let prompt_mel = Tensor::randn(0f32, 1f32, (1, 80, prompt_frames), &device)?;
    let prompt_cond = pipe
        .s2mel
        .length_regulator
        .forward(&spk_w2v, prompt_frames)?;

    let text = Tensor::from_vec(vec![1000u32, 2000, 3000, 4000], (1, 4), &device)?;
    let _ = text;
    // 真实文本走分词器
    let tok = pipe.text_tokenizer(&path)?;
    let (text_ids, lang_id) = pipe.encode_text(&tok, "zh", "你好世界")?;
    println!("text ids: {:?}, lang={lang_id}", text_ids.to_vec2::<u32>()?);
    let opt = GenOptions {
        max_new_tokens: 8,
        ..Default::default()
    };
    let codes = pipe.generate(&text_ids, lang_id, &conds, &opt)?;
    println!("codes: {:?}", codes.dims());
    assert!(codes.dim(1)? > 0 && codes.dim(1)? <= 8);

    let wav = pipe.codes_to_wav(&codes, &prompt_mel, &prompt_cond, &campplus, 2, 0.7)?;
    println!("wav: {:?} @ {} Hz", wav.dims(), pipe.sample_rate);
    assert_eq!(wav.dim(0)?, 1);
    assert_eq!(wav.dim(1)?, 1);
    assert!(wav.dim(2)? > 0);
    let max = wav.abs()?.max_all()?.to_scalar::<f32>()?;
    println!("wav 峰值 {max:.4}（clamp 到 1）");
    assert!(max.is_finite() && max <= 1.0001);
    Ok(())
}

/// 情绪文本编码器（Qwen3-0.6B，只用 hidden states）
#[test]
#[ignore]
fn index_tts_qwen_emotion() -> Result<()> {
    let Some(path) = model_path() else {
        println!("跳过：未设置 AHA_GGUF");
        return Ok(());
    };
    let mut file = File::open(&path)?;
    let lc = gguf_lenient::read(&mut file)?;

    let cfg_bytes = gguf_lenient::embedded_file(&mut file, &lc, "qwen0.6bemo4-merge/config.json")?
        .ok_or_else(|| anyhow::anyhow!("GGUF 里没有 qwen 情绪模型 config.json"))?;
    let cfg: Qwen3Config = serde_json::from_slice(&cfg_bytes)?;
    println!(
        "qwen_emotion cfg: hidden={} layers={} heads={}/{} vocab={} tie={}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.vocab_size,
        cfg.tie_word_embeddings
    );

    let device = aha::utils::get_device(None);
    let gguf = Gguf::new(lc.into_content(), file, device.clone());
    let vb = VarBuilder::from_backend(
        Box::new(GgufVarBuilder::new(gguf, "qwen_emotion")),
        DType::F32,
        device.clone(),
    );
    let mut model = Qwen3Model::new(&cfg, vb, vec![cfg.eos_token_id])?;

    let ids = Tensor::from_vec(vec![100u32, 200, 300, 400, 500, 600], (1, 6), &device)?;
    let hidden = model.forward_hidden_full(Some(&ids), None, 0)?;
    println!(
        "qwen_emotion: ids {:?} -> hidden {:?}",
        ids.dims(),
        hidden.dims()
    );
    assert_eq!(hidden.dims(), &[1, 6, cfg.hidden_size]);
    let max = hidden.abs()?.max_all()?.to_scalar::<f32>()?;
    println!("hidden 峰值 {max:.4}");
    assert!(max.is_finite());
    Ok(())
}

/// s2mel：gpt_layer + length_regulator + U-DiT（一次前向 + 一步 Euler）
#[test]
#[ignore]
fn index_tts_s2mel_forward() -> Result<()> {
    let Some(path) = model_path() else {
        println!("跳过：未设置 AHA_GGUF");
        return Ok(());
    };
    let mut file = File::open(&path)?;
    let lc = gguf_lenient::read(&mut file)?;
    let device = aha::utils::get_device(None);
    let gguf = Gguf::new(lc.into_content(), file, device.clone());
    let vb = VarBuilder::from_backend(
        Box::new(GgufVarBuilder::new(gguf, "s2mel")),
        DType::F32,
        device.clone(),
    );

    // 参数来自 config.yaml：DiT hidden 512 / 8 heads / 13 层、wavenet 8 层、length_regulator 512 通道
    let mut model = S2Mel::new(&vb, 80, 512, 8, 13, 512, 1024, 512, 4, 8, 1280)?;

    // 1) GPT hidden -> 语义特征空间
    let gpt_hidden = Tensor::randn(0f32, 1f32, (1, 24, 1280), &device)?;
    let proj = model.project_gpt(&gpt_hidden)?;
    println!("gpt_layer: {:?} -> {:?}", gpt_hidden.dims(), proj.dims());
    assert_eq!(proj.dims(), &[1, 24, 512 * 2]);

    // 2) length regulator：24 帧 -> 48 帧
    let cond_in = Tensor::randn(0f32, 1f32, (1, 24, 1024), &device)?;
    let cond = model.length_regulator.forward(&cond_in, 48)?;
    println!(
        "length_regulator: {:?} -> {:?}",
        cond_in.dims(),
        cond.dims()
    );
    assert_eq!(cond.dims(), &[1, 48, 512]);

    // 3) 一次 DiT 前向
    let x = Tensor::randn(0f32, 1f32, (1, 80, 48), &device)?;
    let prompt = Tensor::randn(0f32, 1f32, (1, 80, 48), &device)?;
    let style = Tensor::randn(0f32, 1f32, (1, 192), &device)?;
    let t = Tensor::from_vec(vec![0.3f32], (1,), &device)?;
    let mel = model
        .cfm
        .estimator
        .forward(&x, &prompt, &t, &style, &cond)?;
    println!(
        "DiT: x {:?} + cond {:?} -> mel {:?}",
        x.dims(),
        cond.dims(),
        mel.dims()
    );
    assert_eq!(mel.dims(), &[1, 80, 48]);
    let max = mel.abs()?.max_all()?.to_scalar::<f32>()?;
    println!("mel 峰值 {max:.4}");
    assert!(max.is_finite());

    // 4) 一步 Euler（不带 CFG，省时间）
    let mel2 = model.cfm.solve_euler(&cond, &prompt, &style, 1, 0.0, 1.0)?;
    println!("solve_euler(1 step): mel {:?}", mel2.dims());
    assert_eq!(mel2.dims(), &[1, 80, 48]);
    Ok(())
}

/// 情绪条件链路：w2v-bert 特征 -> conformer -> perceiver -> emovec/emo_layer
#[test]
#[ignore]
fn index_tts_emo_conditioner() -> Result<()> {
    let Some(path) = model_path() else {
        println!("跳过：未设置 AHA_GGUF");
        return Ok(());
    };
    let mut file = File::open(&path)?;
    let lc = gguf_lenient::read(&mut file)?;
    let cfg_bytes = gguf_lenient::embedded_file(&mut file, &lc, "config.yaml")?
        .ok_or_else(|| anyhow::anyhow!("GGUF 里没有 config.yaml"))?;
    let cfg = IndexTtsConfig::from_yaml(&cfg_bytes)?;
    let emo_cfg = cfg
        .gpt
        .emo_condition_module
        .clone()
        .ok_or_else(|| anyhow::anyhow!("config.yaml 里没有 emo_condition_module"))?;
    println!(
        "emo_condition_module: out={} linear={} heads={} blocks={} perceiver_mult={}",
        emo_cfg.output_size,
        emo_cfg.linear_units,
        emo_cfg.attention_heads,
        emo_cfg.num_blocks,
        emo_cfg.perceiver_mult
    );

    let device = aha::utils::get_device(None);
    let gguf = Gguf::new(lc.into_content(), file, device.clone());
    let vb = VarBuilder::from_backend(
        Box::new(GgufVarBuilder::new(gguf, "gpt")),
        DType::F32,
        device.clone(),
    );
    let cond = EmoConditioner::new(
        &vb,
        1024,
        emo_cfg.output_size,
        emo_cfg.attention_heads,
        emo_cfg.linear_units,
        emo_cfg.num_blocks,
        1024,
        emo_cfg.perceiver_mult,
        cfg.gpt.model_dim,
    )?;

    let frames = 40;
    let feats = Tensor::randn(0f32, 1f32, (1, frames, 1024), &device)?;
    let emo_vec = cond.forward(&feats)?;
    println!(
        "emo: w2v 特征 {:?} -> conformer+perceiver -> emo_vec {:?}",
        feats.dims(),
        emo_vec.dims()
    );
    assert_eq!(emo_vec.dims(), &[1, cfg.gpt.model_dim]);
    let max = emo_vec.abs()?.max_all()?.to_scalar::<f32>()?;
    println!("emo_vec 绝对值峰值 {max:.4}（应有限且非零）");
    assert!(max.is_finite() && max > 0.0);
    Ok(())
}

/// 用 GGUF 里的权重搭出 IndexTTS 的 GPT 主干（24 层 / 1280 维）
#[test]
#[ignore]
fn index_tts_gpt_trunk() -> Result<()> {
    let Some(path) = model_path() else {
        println!("跳过：未设置 AHA_GGUF");
        return Ok(());
    };
    let mut file = File::open(&path)?;
    let lc = gguf_lenient::read(&mut file)?;

    let cfg_bytes = gguf_lenient::embedded_file(&mut file, &lc, "config.yaml")?
        .ok_or_else(|| anyhow::anyhow!("GGUF 里没有 config.yaml"))?;
    let cfg = IndexTtsConfig::from_yaml(&cfg_bytes)?;
    println!(
        "gpt cfg: dim={} layers={} heads={} text_vocab={} mel_codes={} mel_tokens<={}",
        cfg.gpt.model_dim,
        cfg.gpt.layers,
        cfg.gpt.heads,
        cfg.gpt.number_text_tokens,
        cfg.gpt.number_mel_codes,
        cfg.gpt.max_mel_tokens
    );
    if let Some(ds) = &cfg.dataset
        && let Some(mel) = &ds.mel
    {
        println!(
            "mel frontend: sr={} n_fft={} hop={} n_mels={}",
            ds.sample_rate.unwrap_or(0),
            mel.n_fft,
            mel.hop_length,
            mel.n_mels
        );
    }

    let device = aha::utils::get_device(None);
    let gguf = Gguf::new(lc.into_content(), file, device.clone());
    let vb = VarBuilder::from_backend(
        Box::new(GgufVarBuilder::new(gguf, "gpt")),
        DType::F32,
        device.clone(),
    );
    let mut gpt = IndexTtsGpt::new(&vb, &cfg.gpt)?;

    // 一次前向：8 个 mel code
    let ids = Tensor::from_vec(vec![100u32; 8], (1, 8), &device)?;
    let embeds = gpt.embed_mel(&ids, 0)?;
    let hidden = gpt.forward(&embeds, 0)?;
    let logits = gpt.mel_logits(&hidden)?;
    println!(
        "prefill: embeds {:?} -> hidden {:?} -> mel logits {:?}",
        embeds.dims(),
        hidden.dims(),
        logits.dims()
    );
    assert_eq!(hidden.dims(), &[1, 8, cfg.gpt.model_dim]);
    assert_eq!(logits.dims(), &[1, 8, cfg.gpt.number_mel_codes]);

    // 自回归一步（走 KV cache）
    let next = Tensor::from_vec(vec![200u32], (1, 1), &device)?;
    let embeds = gpt.embed_mel(&next, 8)?;
    let hidden = gpt.forward(&embeds, 8)?;
    println!("decode step: hidden {:?}", hidden.dims());
    assert_eq!(hidden.dims(), &[1, 1, cfg.gpt.model_dim]);

    gpt.clear_kv_cache();
    Ok(())
}

/// 用 GGUF 里的权重直接实例化 aha 现有 `RepCodec`（语义 codec，ConvNeXt/MaskGCT）
#[test]
#[ignore]
fn semantic_codec_forward() -> Result<()> {
    let Some(path) = model_path() else {
        println!("跳过：未设置 AHA_GGUF");
        return Ok(());
    };
    let mut file = File::open(&path)?;
    let lc = gguf_lenient::read(&mut file)?;
    let device = aha::utils::get_device(None);

    let gguf = Gguf::new(lc.into_content(), file, device.clone());
    let vb = VarBuilder::from_backend(
        Box::new(GgufVarBuilder::new(gguf, "semantic_codec")),
        DType::F32,
        device.clone(),
    );
    // 来自内嵌 config.yaml 的 semantic_codec 段；down/up 卷积存在 => downsample_scale=2
    let cfg = SemanticCodec {
        codebook_size: 8192,
        hidden_size: 1024,
        codebook_dim: 8,
        vocos_dim: 384,
        vocos_intermediate_dim: 2048,
        vocos_num_layers: 12,
        num_quantizers: 1,
        downsample_scale: 2,
    };
    let model = RepCodec::new(vb, &cfg)?;

    let frames = 32;
    let xs = Tensor::randn(0f32, 1f32, (1, frames, cfg.hidden_size), &device)?;
    let (rec, indices) = model.forward(&xs)?;
    println!(
        "semantic_codec: {:?} -> 重建 {:?}, 码本索引 {:?}",
        xs.dims(),
        rec.dims(),
        indices.dims()
    );
    assert_eq!(rec.dims(), &[1, frames, cfg.hidden_size]);
    // 码本索引形状是 (bs, num_quantizers, len)
    assert_eq!(indices.dim(indices.rank() - 1)?, frames / 2);

    // 编码路径（真正会被 GPT 用到的方向）
    let (idx, latents) = model.quantize(&xs)?;
    println!(
        "quantize: indices {:?}, latents {:?}",
        idx.dims(),
        latents.dims()
    );
    let max_idx = idx.max_all()?.to_scalar::<u32>()?;
    println!("最大码本索引 {max_idx}（应 < {}）", cfg.codebook_size);
    assert!(max_idx < cfg.codebook_size as u32);
    Ok(())
}

/// 用 GGUF 里的权重直接实例化 aha 现有 `W2VBert2_0Model`（语义特征提取）
#[test]
#[ignore]
fn w2v_bert_forward() -> Result<()> {
    let Some(path) = model_path() else {
        println!("跳过：未设置 AHA_GGUF");
        return Ok(());
    };
    let mut file = File::open(&path)?;
    let lc = gguf_lenient::read(&mut file)?;

    let cfg_bytes = gguf_lenient::embedded_file(&mut file, &lc, "w2v-bert-2.0/config.json")?
        .ok_or_else(|| anyhow::anyhow!("GGUF 里没有 w2v-bert-2.0/config.json"))?;
    let cfg: W2VBert2_0Config = serde_json::from_slice(&cfg_bytes)?;
    println!(
        "w2v-bert cfg: in_dim={} hidden={} layers={} heads={} pos={}",
        cfg.feature_projection_input_dim,
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.position_embeddings_type
    );

    let device = aha::utils::get_device(None);
    let gguf = Gguf::new(lc.into_content(), file, device.clone());
    let vb = VarBuilder::from_backend(
        Box::new(GgufVarBuilder::new(gguf, "wav2vec2bert")),
        DType::F32,
        device.clone(),
    );
    let model = W2VBert2_0Model::new(vb, &cfg)?;

    let frames = 64;
    let feats = Tensor::randn(
        0f32,
        1f32,
        (1, frames, cfg.feature_projection_input_dim),
        &device,
    )?;
    let out = model.forward(&feats, None, None, false)?;
    println!(
        "w2v-bert: feats {:?} -> hidden {:?}",
        feats.dims(),
        out.last_hidden_state.dims()
    );
    assert_eq!(out.last_hidden_state.dims(), &[1, frames, cfg.hidden_size]);
    Ok(())
}

/// 真正把权重读出来（需要完整文件，比较慢）
#[test]
#[ignore]
fn read_all_tensors() -> Result<()> {
    let Some(path) = model_path() else {
        println!("跳过：未设置 AHA_GGUF");
        return Ok(());
    };
    let mut file = File::open(&path)?;
    let lc = gguf_lenient::read(&mut file)?;
    let device = aha::utils::get_device(None);

    let mut done = HashSet::new();
    let mut ok = 0usize;
    let mut bytes = 0u64;
    for (name, info) in lc.content.tensor_infos.iter() {
        if !done.insert(info.offset) {
            continue;
        }
        match info.read(&mut file, lc.content.tensor_data_offset, &device) {
            Ok(q) => {
                ok += 1;
                bytes += q.storage_size_in_bytes() as u64;
            }
            Err(e) => println!("读失败 {name}: {e}"),
        }
    }
    println!(
        "读取完成: {ok} 个张量, 共 {:.2} GB",
        bytes as f64 / 1024.0 / 1024.0 / 1024.0
    );
    assert_eq!(ok, lc.report.tensor_count - lc.report.skipped.len());
    Ok(())
}

/// 用 GGUF 里的权重直接实例化 aha 现有 `BigVGAN` 并跑一次前向（需要完整文件）
#[test]
#[ignore]
fn bigvgan_forward() -> Result<()> {
    let Some(path) = model_path() else {
        println!("跳过：未设置 AHA_GGUF");
        return Ok(());
    };
    let mut file = File::open(&path)?;
    let lc = gguf_lenient::read(&mut file)?;

    // 内嵌的 bigvgan/config.json，字段和 aha 的 BigVGANConfig 一致
    let cfg_bytes = gguf_lenient::embedded_file(&mut file, &lc, "bigvgan/config.json")?
        .ok_or_else(|| anyhow::anyhow!("GGUF 里没有 bigvgan/config.json"))?;
    let cfg: BigVGANConfig = serde_json::from_slice(&cfg_bytes)?;
    println!(
        "bigvgan cfg: num_mels={} ch={} ups={:?}",
        cfg.num_mels, cfg.upsample_initial_channel, cfg.upsample_rates
    );

    let device = aha::utils::get_device(None);
    let gguf = Gguf::new(lc.into_content(), file, device.clone());
    let vb = VarBuilder::from_backend(
        Box::new(GgufVarBuilder::new(gguf, "bigvgan")),
        DType::F32,
        device.clone(),
    );
    let bigvgan = BigVGAN::new(vb, &cfg)?;

    let frames = 50;
    let mel = Tensor::randn(0f32, 1f32, (1, cfg.num_mels, frames), &device)?;
    let wav = bigvgan.forward(&mel)?;
    let expect = frames * cfg.upsample_rates.iter().product::<usize>();
    println!(
        "mel {:?} -> wav {:?}（期望最后一维 {expect}）",
        mel.dims(),
        wav.dims()
    );
    assert_eq!(wav.dim(0)?, 1);
    assert_eq!(wav.dim(1)?, 1);
    assert_eq!(wav.dim(2)?, expect);
    let max = wav.abs()?.max_all()?.to_scalar::<f32>()?;
    println!("输出峰值 {max:.4}（clamp 到 [-1,1]，应 <= 1）");
    assert!(max <= 1.0001);
    Ok(())
}

/// Q4_K 量化权重走 `QMatMul`（F32 conv 之外的另一条路径）
#[test]
#[ignore]
fn qmatmul_smoke() -> Result<()> {
    let Some(path) = model_path() else {
        println!("跳过：未设置 AHA_GGUF");
        return Ok(());
    };
    let mut file = File::open(&path)?;
    let lc = gguf_lenient::read(&mut file)?;
    let device = aha::utils::get_device(None);
    let mut gguf = Gguf::new(lc.into_content(), file, device.clone());

    let name = "gpt/gpt.h.0.attn.c_attn.weight";
    let w = gguf.qmatmul(name)?;
    // 注意：GPT-2 的 `Conv1D` 权重是 (in, out) 存的，和 nn.Linear 的 (out, in) 相反。
    // 这里 GGUF 维度翻转后正好是 PyTorch 的 (in=1280, out=3840)。
    let (rows, cols) = match gguf.tensor(name)?.shape().dims() {
        [a, b] => (*a, *b),
        d => anyhow::bail!("意外的维度 {d:?}"),
    };
    println!("{name}: [{rows}, {cols}]（Q4_K，量化 QMatMul）");

    let xs = Tensor::randn(0f32, 1f32, (1, cols), &device)?;
    let ys = w.forward(&xs)?;
    println!(
        "matmul: {:?} x [{rows},{cols}] -> {:?}",
        xs.dims(),
        ys.dims()
    );
    assert_eq!(ys.dims(), &[1, rows]);

    // 和 dequantize 后手算的结果对一下，确认量化路径没走偏。
    // 注意 candle 的 QMatMul 在 CPU 上走 f16 通路（输入 cast 到 f16 + f16 累加），
    // 所以这里只能看相对误差，不能要求逐位一致。
    let wd = gguf.get_dequantized(name)?;
    let expect = xs.matmul(&wd.t()?.contiguous()?)?;
    let diff = (&ys - &expect)?.abs()?.max_all()?.to_scalar::<f32>()?;
    let scale = expect.abs()?.max_all()?.to_scalar::<f32>()?;
    println!(
        "与反量化后 f32 matmul 对比: 最大偏差 {diff:.5}, 输出峰值 {scale:.3}, 相对 {:.3}%",
        diff / scale.max(1e-6) * 100.0
    );
    assert!(
        diff / scale.max(1e-6) < 0.05,
        "相对偏差过大: {diff} / {scale}"
    );
    Ok(())
}
