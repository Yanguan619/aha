//! GPT-SoVITS P3（T2S AR）自检：能加载、能增量解码、greedy 结果可复现。
//!
//! 注意：这里**不验证音质**，也不比对 Python 的 token 序列——采样用的是不同的随机源，
//! 只有 `greedy` 模式才可能逐 token 对齐。等 `s1` 的参考输出备好后，再补一个
//! 「同 greedy 前缀一致」的对拍测试。
//!
//! 没有设置 `AHA_GPT_SOVITS_DIR` 时跳过。

use std::path::PathBuf;

use aha::models::gpt_sovits::{GenOptions, RobertaEncoder, Text2Semantic, WordPieceTokenizer, ZhFrontend};
use anyhow::Result;
use candle_core::{DType, Device};

fn model_dir() -> Option<PathBuf> {
    let dir = std::env::var("AHA_GPT_SOVITS_DIR").ok()?;
    let path = PathBuf::from(dir);
    path.join("gpt.safetensors").is_file().then_some(path)
}

#[test]
fn t2s_greedy_decode_is_reproducible() -> Result<()> {
    let Some(dir) = model_dir() else {
        println!("跳过：未设置 AHA_GPT_SOVITS_DIR");
        return Ok(());
    };
    let device = Device::Cpu;

    let frontend = ZhFrontend::load(&dir)?;
    let tokenizer = WordPieceTokenizer::load(dir.join("bert_tokenizer/tokenizer.json"))?;
    let bert = RobertaEncoder::load(&dir, &device, DType::F32)?;
    let t2s = Text2Semantic::load(&dir, &device, DType::F32)?;

    assert_eq!(t2s.cfg.n_layer, 24);
    assert_eq!(t2s.cfg.embedding_dim, 512);
    assert_eq!(t2s.cfg.eos, 1024);

    let g2p = frontend.g2p("你好，很高兴认识你。")?;
    let ids = frontend.to_ids(&g2p.phones);
    let text_ids: Vec<u32> = ids;
    let bert_feat = bert.phone_level_feature(&g2p.norm_text, &tokenizer, &g2p.word2ph)?;
    assert_eq!(bert_feat.dims(), &[1024, text_ids.len()]);

    // 提示语义 token 用占位值（真实的要从参考音频经 HuBERT + quantizer 得到）
    let prompt: Vec<u32> = (0..20).map(|i| 100 + i as u32).collect();

    let opts = GenOptions {
        greedy: true,
        max_new_tokens: 24,
        early_stop_num: None,
        ..Default::default()
    };

    let first = t2s.infer(&text_ids, &bert_feat, &prompt, &opts)?;
    let second = t2s.infer(&text_ids, &bert_feat, &prompt, &opts)?;

    println!(
        "text={} phones, prompt={} codes, 生成 {} 个语义 token: {:?}",
        text_ids.len(),
        prompt.len(),
        first.len(),
        first
    );
    assert_eq!(first, second, "greedy 两次结果不一致");
    assert!(first.len() <= opts.max_new_tokens);
    assert!(
        first.iter().all(|t| (*t as usize) < 1024),
        "生成的 token 应该落在 [0, 1024)"
    );
    Ok(())
}

#[test]
fn t2s_handles_empty_prompt() -> Result<()> {
    let Some(dir) = model_dir() else {
        println!("跳过：未设置 AHA_GPT_SOVITS_DIR");
        return Ok(());
    };
    let device = Device::Cpu;
    let frontend = ZhFrontend::load(&dir)?;
    let tokenizer = WordPieceTokenizer::load(dir.join("bert_tokenizer/tokenizer.json"))?;
    let bert = RobertaEncoder::load(&dir, &device, DType::F32)?;
    let t2s = Text2Semantic::load(&dir, &device, DType::F32)?;

    let g2p = frontend.g2p("你好。")?;
    let text_ids = frontend.to_ids(&g2p.phones);
    let bert_feat = bert.phone_level_feature(&g2p.norm_text, &tokenizer, &g2p.word2ph)?;

    let opts = GenOptions {
        greedy: true,
        max_new_tokens: 8,
        ..Default::default()
    };
    // ref_free 模式：不传参考音频
    let out = t2s.infer(&text_ids, &bert_feat, &[], &opts)?;
    println!("ref_free 生成 {} 个 token", out.len());
    Ok(())
}
