//! AR 的 CUDA graph 路径必须和 eager 路径**逐 token 一致**（greedy 下完全可复现）。
//!
//! 抓图的坑都在 `ar_graph.rs` 顶上：kv 定长、`scatter_set` 原位写、mask 要包含刚写的那格，
//! 以及"整块拷贝的索引不能全 0"。这个测试是那几处逻辑的守门员。
//!
//! 没有 `AHA_GPT_SOVITS_DIR` 或没有 cuda 时跳过。

use std::path::PathBuf;
use std::time::Instant;

use aha::models::gpt_sovits::{GenOptions, RobertaEncoder, Text2Semantic, WordPieceTokenizer, ZhFrontend};
use anyhow::Result;
use candle_core::{DType, Device};

fn model_dir() -> Option<PathBuf> {
    let dir = std::env::var("AHA_GPT_SOVITS_DIR").ok()?;
    let path = PathBuf::from(dir);
    path.join("gpt.safetensors").is_file().then_some(path)
}

#[test]
fn graph_decode_matches_eager() -> Result<()> {
    let Some(dir) = model_dir() else {
        println!("跳过：未设置 AHA_GPT_SOVITS_DIR");
        return Ok(());
    };
    let Ok(device) = Device::new_cuda(0) else {
        println!("跳过：没有可用的 CUDA 设备");
        return Ok(());
    };

    let frontend = ZhFrontend::load(&dir)?;
    let tokenizer = WordPieceTokenizer::load(dir.join("bert_tokenizer/tokenizer.json"))?;
    let bert = RobertaEncoder::load(&dir, &device, DType::F32)?;
    let t2s = Text2Semantic::load(&dir, &device, DType::F32)?;

    let g2p = frontend.g2p("今天天气真不错，我们出去玩吧。")?;
    let ids = frontend.to_ids(&g2p.phones);
    let bert_feat = bert.phone_level_feature(&g2p.norm_text, &tokenizer, &g2p.word2ph)?;
    // 参考音的语义 codes 只是上下文，这里用固定序列即可（不影响两条路径的一致性）
    let prompt: Vec<u32> = (0..116).map(|i| (i * 13 % 1023) as u32).collect();

    let opts = GenOptions {
        greedy: true,
        seed: 42,
        early_stop_num: Some(ids.len() * 12 + 48),
        ..Default::default()
    };

    unsafe { std::env::set_var("AHA_GSV_GRAPH", "0") };
    let t = Instant::now();
    let eager = t2s.infer(&ids, &bert_feat, &prompt, &opts)?;
    let eager_ms = t.elapsed().as_secs_f64() * 1000.0;

    unsafe { std::env::remove_var("AHA_GSV_GRAPH") };
    let _ = t2s.infer(&ids, &bert_feat, &prompt, &opts)?; // 首次：抓图
    let t = Instant::now();
    let graph = t2s.infer(&ids, &bert_feat, &prompt, &opts)?;
    let graph_ms = t.elapsed().as_secs_f64() * 1000.0;

    assert_eq!(t2s.graph_count(), 1, "图没有被捕获/使用");
    println!(
        "eager={eager_ms:.0}ms ({:.1}ms/token)  graph={graph_ms:.0}ms ({:.1}ms/token)  tokens={}",
        eager_ms / eager.len() as f64,
        graph_ms / graph.len() as f64,
        eager.len()
    );
    assert_eq!(eager, graph, "图路径和 eager 路径的 token 不一致");
    Ok(())
}
