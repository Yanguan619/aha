//! GPT-SoVITS 中文前端对拍：Rust 输出 vs Python 参考实现。
//!
//! 参考数据由 `scripts/gpt_sovits/reference_frontend.py` 生成（复用 GPT-SoVITS 仓库里的
//! `text/chinese2.py` + `text/tone_sandhi.py`，只把 `jieba_fast` 换成 `jieba`、G2PW 换成
//! pypinyin 兜底）：
//!
//! ```bash
//! uv pip install pypinyin cn2an jieba
//! python scripts/gpt_sovits/reference_frontend.py \
//!     --gsv-root D:/GPT-SoVITS \
//!     --jsonl tests/gpt_sovits_data/corpus_zh.txt \
//!     --out tests/gpt_sovits_data/reference_zh.jsonl
//! ```
//!
//! 运行：
//!
//! ```bash
//! AHA_GPT_SOVITS_DIR=D:/model_download/gpt-sovits-v2pro-zh \
//!   cargo test --features gpt-sovits --test test_gpt_sovits_frontend -- --nocapture
//! ```
//!
//! 没设置 `AHA_GPT_SOVITS_DIR` 时会跳过（不失败）。

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use aha::models::gpt_sovits::ZhFrontend;
use anyhow::{Context, Result, bail};
use serde::Deserialize;

#[derive(Deserialize)]
struct ReferenceRecord {
    text: String,
    phones: Vec<String>,
    word2ph: Vec<usize>,
}

fn model_dir() -> Option<PathBuf> {
    let dir = std::env::var("AHA_GPT_SOVITS_DIR").ok()?;
    let path = PathBuf::from(dir);
    path.join("frontend_zh.json").is_file().then_some(path)
}

fn load_reference(path: &Path) -> Result<Vec<ReferenceRecord>> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("读取 {}", path.display()))?;
    content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).with_context(|| format!("解析 {line}")))
        .collect()
}

#[test]
fn zh_frontend_matches_reference() -> Result<()> {
    let Some(dir) = model_dir() else {
        println!("跳过：未设置 AHA_GPT_SOVITS_DIR（或目录里没有 frontend_zh.json）");
        return Ok(());
    };

    let frontend = ZhFrontend::load(&dir)?;
    let records = load_reference(Path::new("tests/gpt_sovits_data/reference_zh.jsonl"))?;
    if records.is_empty() {
        bail!("参考语料为空");
    }

    // 导出的符号表必须与权重里 embedding 的行数一致
    assert_eq!(frontend.num_symbols(), 732, "symbols.json 数量不对");

    let mut failures = Vec::new();
    let mut total_phones = 0usize;

    for record in &records {
        let out = frontend.g2p(&record.text)?;
        total_phones += record.phones.len();

        // 与 Python 的两个不变量保持一致
        assert_eq!(
            out.word2ph.iter().sum::<usize>(),
            out.phones.len(),
            "{}: sum(word2ph) != len(phones)",
            record.text
        );
        assert_eq!(
            out.word2ph.len(),
            out.norm_text.chars().count(),
            "{}: len(word2ph) != len(norm_text)",
            record.text
        );

        let ids = frontend.to_ids(&out.phones);
        assert_eq!(ids.len(), out.phones.len());

        let phones_ok = out.phones == record.phones;
        let w2p_ok = out.word2ph == record.word2ph;
        if !phones_ok || !w2p_ok {
            let diff = (0..record.phones.len().max(out.phones.len()))
                .filter(|i| record.phones.get(*i) != out.phones.get(*i))
                .count();
            failures.push(format!(
                "{}\n  参考: {}\n  Rust: {}\n  差异: {diff} 个 phone，word2ph {}",
                record.text,
                record.phones.join(" "),
                out.phones.join(" "),
                if w2p_ok { "一致" } else { "不一致" }
            ));
        }
    }

    if !failures.is_empty() {
        bail!(
            "{} / {} 句与参考实现不一致：\n{}",
            failures.len(),
            records.len(),
            failures.join("\n")
        );
    }

    println!("✓ {} 句、{total_phones} 个 phone 与 Python 参考逐元素一致", records.len());
    Ok(())
}

/// 同一个前端反复调用应稳定（防止 HashMap 迭代顺序之类的不确定性混进来）。
#[test]
fn zh_frontend_is_deterministic() -> Result<()> {
    let Some(dir) = model_dir() else {
        return Ok(());
    };
    let frontend = ZhFrontend::load(&dir)?;
    let text = "这个西瓜有三斤半重，我们分着吃吧。";
    let first = frontend.g2p(text)?;
    let seen: HashMap<Vec<String>, usize> = HashMap::new();
    drop(seen);
    for _ in 0..5 {
        let again = frontend.g2p(text)?;
        assert_eq!(first.phones, again.phones);
        assert_eq!(first.word2ph, again.word2ph);
        assert_eq!(first.norm_text, again.norm_text);
    }
    Ok(())
}
