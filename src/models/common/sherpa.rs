//! 桌宠委托模式下的大模型权重下载器（SenseVoice / Kokoro）。
//!
//! 模型权重**不进仓库、不提前下**，首次启动由服务自己按文件列表从 HF 镜像拉
//! （移植自旧 asr-route / tts-route 的 `model.rs`，见 [`ensure_downloaded`]）。
//!
//! 这里只放"按仓库拉文件"的公共逻辑，具体推理实现分别在
//! [`crate::models::sense_voice`] 和 [`crate::models::kokoro`]。

use std::{
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use serde::Deserialize;

/// 两个 sherpa 模型各自的仓库与必填文件。
#[derive(Clone, Copy)]
struct SherpaSpec {
    repo: &'static str,
    /// 少了任何一个都跑不起来（`dict/` 里的 jieba 词典算一个）。
    required: &'static [&'static str],
    /// 默认跳过这些路径前缀（跟推理无关或有大文件）；Kokoro 的 espeak 数据
    /// 从 2.0.0 起被 Rust 绑定忽略，下它是纯浪费。
    skip_prefixes: &'static [&'static str],
}

const SENSE_VOICE: SherpaSpec = SherpaSpec {
    repo: "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
    required: &["model.int8.onnx", "tokens.txt"],
    skip_prefixes: &[],
};

const KOKORO: SherpaSpec = SherpaSpec {
    repo: "csukuangfj/kokoro-multi-lang-v1_1",
    required: &[
        "model.onnx",
        "voices.bin",
        "tokens.txt",
        "lexicon-us-en.txt",
        "lexicon-zh.txt",
        "dict/jieba.dict.utf8",
    ],
    skip_prefixes: &["espeak-ng-data/"],
};

/// 全部文件齐了才写这个；下次启动凭它跳过联网。
const MARKER: &str = ".ready";
/// 小于这个字节数的文件（tokens.txt 之类）不核大小——它们本来就只有几百字节。
const TINY: u64 = 4096;

#[derive(Deserialize)]
struct FileEntry {
    #[serde(rename = "type")]
    kind: String,
    path: String,
    size: Option<u64>,
}

/// 确保 `dir` 里有可用的模型文件；没有就按文件列表从 HF 镜像拉。
///
/// - 已经齐了就直接用，**不联网、不重复下**（`.ready` 标记 + 逐个文件核大小）；
/// - 缺的才从 HF 镜像拉，流式写 `.part` 再原子改名（中断不会留下"看起来存好了"的半截模型）；
/// - 换源用 `HF_ENDPOINT`（例如 `https://huggingface.co`）。
pub fn ensure_downloaded(
    dir: &str,
    model: crate::models::common::model_mapping::WhichModel,
) -> Result<()> {
    let spec = match model {
        #[cfg(feature = "sherpa")]
        crate::models::common::model_mapping::WhichModel::SenseVoice => SENSE_VOICE,
        #[cfg(feature = "sherpa")]
        crate::models::common::model_mapping::WhichModel::Kokoro => KOKORO,
        _ => bail!("不是 sherpa 模型，没有可下载的文件列表"),
    };
    ensure(dir, spec)
}

fn ensure(dir: &str, spec: SherpaSpec) -> Result<()> {
    let dir = Path::new(dir);
    fs::create_dir_all(dir).with_context(|| format!("建不了模型目录 {}", dir.display()))?;

    if ready(dir, spec) {
        tracing::info!(dir = %dir.display(), "模型已就绪（跳过下载）");
        return Ok(());
    }

    let files = fetch_file_list(&spec)?;
    let total_bytes: u64 = files.iter().map(|f| f.size.unwrap_or(0)).sum();
    tracing::info!(
        files = files.len(),
        mb = total_bytes / 1024 / 1024,
        "首次拉取 {} 模型（断线会从缺的文件继续）",
        spec.repo
    );

    let mut done = 0usize;
    for file in &files {
        let dst = dir.join(&file.path);
        if usable(&dst, file.size.unwrap_or(0)) {
            done += 1;
            continue;
        }
        download_file(file, &dst, &spec)?;
        done += 1;
        // 大文件（>1MB）下完报一次；小文件攒 50 个报一次，别把日志刷爆
        if file.size.unwrap_or(0) > 1 << 20 || done.is_multiple_of(50) {
            tracing::info!(path = %file.path, done, total = files.len(), "下载中");
        }
    }

    // 再核一遍必填项：上面任何一个失败都会 bail，但列表里少了文件不会
    for name in spec.required {
        let path = dir.join(name);
        if !path.exists() {
            bail!(
                "{} 里缺少 {name}（模型仓库的文件列表变了？）",
                dir.display()
            );
        }
    }
    fs::write(dir.join(MARKER), spec.repo).context("写 .ready 失败")?;
    tracing::info!(dir = %dir.display(), "模型文件就绪");
    Ok(())
}

/// 已经完整下过一遍？只核必填项——**不依赖 `.ready` 标记**：
/// 模型可能是 Python 那条退回路径下的（它不写标记），文件齐了就不该再联网。
fn ready(dir: &Path, spec: SherpaSpec) -> bool {
    spec.required.iter().all(|name| {
        fs::metadata(dir.join(name))
            .map(|m| m.is_file() && m.len() > 0)
            .unwrap_or(false)
    })
}

/// 问 HF 镜像要一份递归文件列表（`/api/models/<repo>/tree/main?recursive=1`）。
fn fetch_file_list(spec: &SherpaSpec) -> Result<Vec<FileEntry>> {
    let url = format!(
        "{}/api/models/{}/tree/main?recursive=1",
        hf_base(),
        spec.repo
    );
    let client = client()?;
    let response = client
        .get(&url)
        .send()
        .with_context(|| format!("取文件列表失败：{url}"))?;
    if !response.status().is_success() {
        bail!("取文件列表失败：{url} 返回 HTTP {}", response.status());
    }
    let items: Vec<FileEntry> = response.json().context("文件列表不是合法 JSON")?;
    let files: Vec<FileEntry> = items
        .into_iter()
        .filter(|item| item.kind == "file")
        .filter(|item| {
            // 这两个跟推理无关，少下两个文件
            !matches!(item.path.as_str(), ".gitattributes" | "README.md")
        })
        .filter(|item| {
            !spec
                .skip_prefixes
                .iter()
                .any(|prefix| item.path.starts_with(prefix))
        })
        .collect();
    if files.is_empty() {
        bail!("文件列表是空的：{url}");
    }
    Ok(files)
}

fn usable(path: &Path, size: u64) -> bool {
    match fs::metadata(path) {
        Ok(meta) if meta.is_file() => meta.len() > 0 && (size < TINY || meta.len() == size),
        _ => false,
    }
}

fn download_file(file: &FileEntry, dst: &Path, spec: &SherpaSpec) -> Result<()> {
    let url = format!("{}/{}/resolve/main/{}", hf_base(), spec.repo, file.path);
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent).with_context(|| format!("建不了目录 {}", parent.display()))?;
    }

    let mut response = client()?
        .get(&url)
        .send()
        .with_context(|| format!("下载失败：{url}"))?;
    if !response.status().is_success() {
        bail!("下载失败：{url} 返回 HTTP {}", response.status());
    }

    let part = with_suffix(dst, ".part");
    let mut out = File::create(&part).with_context(|| format!("写不了 {}", part.display()))?;
    let mut buffer = vec![0u8; 1 << 20];
    let mut got: u64 = 0;
    loop {
        let read = response.read(&mut buffer).context("下载中断")?;
        if read == 0 {
            break;
        }
        out.write_all(&buffer[..read]).context("写模型文件失败")?;
        got += read as u64;
    }
    out.flush().context("刷盘失败")?;
    drop(out);

    // 大小对不上就是没下完（CDN 偶尔会截断）：别把半截文件改名成正式文件
    if let Some(expected) = file.size
        && expected >= TINY
        && got != expected
    {
        bail!("下载不完整：{} 得到 {got} 字节，期望 {expected}", file.path);
    }
    fs::rename(&part, dst).with_context(|| format!("改名失败：{}", part.display()))?;
    Ok(())
}

fn hf_base() -> String {
    std::env::var("HF_ENDPOINT")
        .unwrap_or_else(|_| "https://hf-mirror.com".to_owned())
        .trim_end_matches('/')
        .to_owned()
}

fn client() -> Result<reqwest::blocking::Client> {
    reqwest::blocking::Client::builder()
        // 30 分钟是"整条请求"的上限：model.onnx（325MB）能在 180KB/s 的线下完
        .timeout(std::time::Duration::from_secs(1800))
        .connect_timeout(std::time::Duration::from_secs(30))
        .build()
        .context("建不了 HTTP 客户端")
}

fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(suffix);
    path.with_file_name(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specs_carry_required_files() {
        assert!(SENSE_VOICE.required.contains(&"model.int8.onnx"));
        assert!(KOKORO.required.contains(&"dict/jieba.dict.utf8"));
    }

    #[test]
    fn usable_treats_tiny_files_leniently() {
        let dir = std::env::temp_dir();
        let path = dir.join("aha_sherpa_tiny_test.txt");
        std::fs::write(&path, b"hi").unwrap();
        // 远端大小 < TINY（tokens.txt 之类）：不核实际字节数，存在就算可用
        assert!(usable(&path, 100));
        // 远端大小 >= TINY（模型文件）：必须字节数完全一致，半截文件不算
        assert!(!usable(&path, 12345));
        std::fs::remove_file(&path).unwrap();
    }
}
