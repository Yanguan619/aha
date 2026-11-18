//! 桌宠委托模式下的大模型权重下载器（SenseVoice / Kokoro）。
//!
//! 模型权重**不进仓库、不提前下**，首次启动由服务自己按文件列表从 ModelScope
//! 镜像拉（复用 [`crate::utils::download_files_to`]，移植自旧 asr-route /
//! tts-route 的 `model.rs`，见 [`ensure_downloaded`]）。
//!
//! 官方权重在 HF 的 `csukuangfj/...` 下，这里换成社区搬到 ModelScope 的镜像：
//! 国内直连、不用 `HF_ENDPOINT`，也能跟其它模型共用同一个下载器/缓存目录。
//!
//! 这里只放"按仓库拉文件"的公共逻辑，具体推理实现分别在
//! [`crate::models::sense_voice`] 和 [`crate::models::kokoro`]。

use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::utils::{DownloadFilter, download_files_to};

/// 两个 sherpa 模型各自的 ModelScope 镜像仓库与必填文件。
#[derive(Clone, Copy)]
struct SherpaSpec {
    /// ModelScope 仓库 id（HF 上的 `csukuangfj/...` 在 ModelScope 没有同名镜像，
    /// 直接用会 404）。
    repo: &'static str,
    /// 少了任何一个都跑不起来（`dict/` 里的 jieba 词典算一个）。
    ///
    /// 下载时**只**拉这些文件：镜像里常多放一份 fp32 权重（SenseVoice 的
    /// `model.onnx` 有 900MB+）或 espeak 数据，下了也用不上。
    required: &'static [&'static str],
}

// 两个 spec 各自跟着自己的模型 feature 走：只编了 sherpa-asr 时不需要 Kokoro 的
// 仓库/文件清单。
#[cfg(feature = "sherpa-asr")]
const SENSE_VOICE: SherpaSpec = SherpaSpec {
    repo: crate::models::mirrors::SENSE_VOICE_REPO,
    required: crate::models::mirrors::SENSE_VOICE_FILES,
};

#[cfg(feature = "sherpa-tts")]
const KOKORO: SherpaSpec = SherpaSpec {
    repo: crate::models::mirrors::KOKORO_REPO,
    required: crate::models::mirrors::KOKORO_FILES,
};

/// 全部文件齐了才写这个；下次启动凭它跳过联网。
const MARKER: &str = ".ready";

/// 确保 `dir` 里有可用的模型文件；没有就从 ModelScope 镜像拉。
///
/// - 已经齐了就直接用，**不联网、不重复下**（逐个文件核大小）；
/// - 缺的才从 ModelScope 下，只下 [`SherpaSpec::required`] 里的文件；
/// - 下完再核一遍必填项并写 `.ready` 标记（只记来源仓库，方便排查；下次启动
///   判定是否就绪只看必填文件本身）。
pub async fn ensure_downloaded(
    dir: &str,
    model: crate::models::common::model_mapping::WhichModel,
) -> Result<()> {
    let spec = match model {
        #[cfg(feature = "sherpa-asr")]
        crate::models::common::model_mapping::WhichModel::SenseVoice => SENSE_VOICE,
        #[cfg(feature = "sherpa-tts")]
        crate::models::common::model_mapping::WhichModel::Kokoro => KOKORO,
        _ => bail!("不是 sherpa 模型，没有可下载的文件列表"),
    };
    ensure(dir, spec).await
}

async fn ensure(dir: &str, spec: SherpaSpec) -> Result<()> {
    let dir = Path::new(dir);
    fs::create_dir_all(dir).with_context(|| format!("建不了模型目录 {}", dir.display()))?;

    if ready(dir, spec) {
        tracing::info!(dir = %dir.display(), "模型已就绪（跳过下载）");
        return Ok(());
    }

    tracing::info!(
        repo = spec.repo,
        files = spec.required.len(),
        "首次从 ModelScope 拉取 {} 模型",
        spec.repo
    );
    download_files_to(
        spec.repo,
        dir,
        3,
        DownloadFilter {
            only: Some(spec.required),
            skip_prefixes: &[],
        },
    )
    .await
    .with_context(|| format!("从 ModelScope 下载 {} 失败", spec.repo))?;

    // 再核一遍必填项：下载器只按列表拉，列表里少了文件不会报
    for name in spec.required {
        let path = dir.join(name);
        if !path.exists() {
            bail!(
                "{} 里缺少 {name}（ModelScope 仓库的文件列表变了？）",
                dir.display()
            );
        }
    }
    fs::write(dir.join(MARKER), spec.repo).context("写 .ready 失败")?;
    tracing::info!(dir = %dir.display(), "模型文件就绪");
    Ok(())
}

/// 已经完整下过一遍？只核必填项——**不依赖 `.ready` 标记**：
/// 模型可能是手动放进目录的（不写标记），文件齐了就不该再联网。
fn ready(dir: &Path, spec: SherpaSpec) -> bool {
    spec.required.iter().all(|name| {
        fs::metadata(dir.join(name))
            .map(|m| m.is_file() && m.len() > 0)
            .unwrap_or(false)
    })
}

/// ONNX Runtime 的 execution provider（默认不指定，交给 sherpa / ORT 兜底到 CPU）。
///
/// 只影响 ASR/TTS 这类 sherpa 模型，取值跟 sherpa-onnx 一致：
/// `cpu` / `cuda` / `coreml` / `trt` / `rknn` / `qnn` / `openvino` / `nnapi`。
///
/// 优先级：`--provider <p>` > 环境变量 `AHA_PROVIDER` > 不指定（CPU）。
/// `--device npu` 会在这里被翻成探测到的 NPU provider（见 [`crate::utils::npu`]）。
///
/// Android 上想上 NPU 用 `--device npu`（= 探测到 QNN 就用 QNN）或直接
/// `--provider qnn`；NNAPI 在 Android 15+ 基本失效，只在老机器上还有意义。
/// 预编译的 `libonnxruntime.so` 不带对应 EP 时，sherpa 会打印
/// `Failed to create ...` / `Fallback to cpu` 然后照常跑——`aha npu` 可以提前告诉
/// 你环境里到底有没有 QNN 运行时。
pub fn onnx_provider() -> Option<String> {
    crate::utils::npu::effective_provider()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specs_carry_required_files() {
        #[cfg(feature = "sherpa-asr")]
        assert!(SENSE_VOICE.required.contains(&"model.int8.onnx"));
        #[cfg(feature = "sherpa-tts")]
        assert!(KOKORO.required.contains(&"dict/jieba.dict.utf8"));
    }

    #[test]
    fn specs_download_from_modelscope_mirrors() {
        // HF 上的 `csukuangfj/...` 在 ModelScope 没有同名镜像（直接查会 404），
        // 别再改回去——否则国内直连下不下来。
        #[cfg(feature = "sherpa-asr")]
        assert!(SENSE_VOICE.repo.starts_with("WEAAEW/"));
        #[cfg(feature = "sherpa-tts")]
        assert!(KOKORO.repo.starts_with("journey0ad/"));
    }
}
