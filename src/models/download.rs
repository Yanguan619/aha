//! 面向 FFI / 外部宿主的"给我模型 id，把权重拉下来"入口。
//!
//! CLI 里那套下载逻辑散在 `pet_model_or_download` / `gguf_or_download` 里，跟 clap 的
//! `WhichModel` 枚举绑得太死。这里抽出一个只吃字符串的版本：宿主（Android 的 JNI 壳、
//! 桌宠主进程）不需要知道 `WhichModel` 有哪些变体，只要传模型 id。
//!
//! 仓库 id 和文件清单来自 [`crate::models::mirrors`]，下载本身复用
//! [`crate::utils::download_files_to_with_ctl`]（进度回调 + 取消都走 `DownloadCtl`）。

use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow, bail};

use crate::models::mirrors::{self, Mirror};
use crate::utils::{
    DownloadCtl, DownloadFilter, DownloadSummary, download_files_to_with_ctl, get_default_save_dir,
};

/// 已就位的 GGUF 权重：`model` 一定存在，`mmproj` 是可选的投影文件（纯文本模型为 `None`）。
pub struct GgufWeights {
    pub model: PathBuf,
    pub mmproj: Option<PathBuf>,
}

/// 确保 `dir` 里有 GGUF 权重，返回可加载的路径。
///
/// 镜像表列了文件（qwen3.5-gguf）时以**第一个文件为主权重**：它在就不联网；列表里其余
/// 形如 `mmproj*` 的文件当作可选投影。镜像表没列文件（Hy-MT2 整仓）时扫目录里的
/// `.gguf`，优先 `1.25` 量化。
pub async fn ensure_gguf(model_id: &str, dir: &Path, max_retries: u32) -> Result<GgufWeights> {
    match mirror_entry(model_id).1 {
        Some(files) => {
            let model = dir.join(files[0]);
            if !model.is_file() {
                tracing::info!(model_id, dir = %dir.display(), "从 ModelScope 下载 GGUF 权重");
                download_into(model_id, dir, max_retries, DownloadCtl::default()).await?;
            }
            if !model.is_file() {
                bail!("GGUF 权重不完整，缺：{}", model.display());
            }
            let mmproj = files
                .iter()
                .skip(1)
                .find(|file| file.starts_with("mmproj"))
                .map(|file| dir.join(file))
                .filter(|path| path.is_file());
            Ok(GgufWeights { model, mmproj })
        }
        None => {
            let mut model = find_gguf(dir)?;
            if model.is_none() {
                tracing::info!(model_id, dir = %dir.display(), "从 ModelScope 下载 GGUF 权重");
                download_into(model_id, dir, max_retries, DownloadCtl::default()).await?;
                model = find_gguf(dir)?;
            }
            let model = model.ok_or_else(|| anyhow!("{} 里没有 .gguf 权重文件", dir.display()))?;
            Ok(GgufWeights {
                model,
                mmproj: None,
            })
        }
    }
}

/// 同步版 [`ensure_gguf`]：`aha run` 本身是同步的，在独立线程里跑一个运行时，
/// 避免"runtime within runtime"。
pub fn ensure_gguf_blocking(model_id: &str, dir: &str) -> Result<(String, Option<String>)> {
    let model_id = model_id.to_string();
    let dir = PathBuf::from(dir);
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        rt.block_on(ensure_gguf(&model_id, &dir, 3))
    })
    .join()
    .map_err(|_| anyhow!("GGUF 下载线程 panic"))?
    .map(|weights| {
        (
            weights.model.display().to_string(),
            weights.mmproj.map(|path| path.display().to_string()),
        )
    })
}

/// 确保镜像表登记的文件都在（缺就整份清单重下）。
pub async fn ensure_files(model_id: &str, dir: &Path, max_retries: u32) -> Result<()> {
    let files = mirror_entry(model_id)
        .1
        .ok_or_else(|| anyhow!("{model_id} 没有在 mirrors 里登记文件清单"))?;
    if files.iter().all(|file| dir.join(file).is_file()) {
        return Ok(());
    }
    tracing::info!(model_id, dir = %dir.display(), "从 ModelScope 下载模型权重");
    download_into(model_id, dir, max_retries, DownloadCtl::default()).await?;
    let missing: Vec<&str> = files
        .iter()
        .copied()
        .filter(|file| !dir.join(file).is_file())
        .collect();
    if !missing.is_empty() {
        bail!("{model_id} 下载完仍缺文件：{}", missing.join(", "));
    }
    Ok(())
}

/// 确保 `dir` 是一份"能加载的权重目录"：`looks_like` 认了就直接返回，
/// 否则按镜像表整份下载后再复核一遍（给"必须凑齐一组文件"的模型用，如 laya）。
///
/// `looks_like` 由模型模块提供——它只回答"这堆文件像不像我的权重"，不碰下载细节。
pub async fn ensure_dir(
    model_id: &str,
    dir: &Path,
    max_retries: u32,
    looks_like: impl Fn(&Path) -> bool,
) -> Result<()> {
    if looks_like(dir) {
        return Ok(());
    }
    tracing::info!(model_id, dir = %dir.display(), "从 ModelScope 下载模型权重");
    download_into(model_id, dir, max_retries, DownloadCtl::default()).await?;
    if !looks_like(dir) {
        bail!("{model_id} 下载完仍不像可加载的权重目录：{}", dir.display());
    }
    Ok(())
}

/// 在 `dir` 里找 GGUF 文件；优先带 `1.25` 的那个，其次任意 `.gguf`。
fn find_gguf(dir: &Path) -> Result<Option<PathBuf>> {
    if !dir.is_dir() {
        return Ok(None);
    }
    let mut names: Vec<String> = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.to_lowercase().ends_with(".gguf") {
            names.push(name);
        }
    }
    names.sort();
    Ok(pick_gguf_file(&names).map(|name| dir.join(name)))
}

/// 从一组文件名里挑权重：先找 `1.25` 量化，再退化为第一个 `.gguf`。
fn pick_gguf_file(names: &[String]) -> Option<&String> {
    names
        .iter()
        .find(|name| name.to_lowercase().ends_with(".gguf") && name.contains("1.25"))
        .or_else(|| {
            names
                .iter()
                .find(|name| name.to_lowercase().ends_with(".gguf"))
        })
}

/// 模型 id 在镜像表里的 `(仓库, 文件清单)`；不在表里就把它当仓库 id（文件清单为 `None`，
/// 整仓下载）。
pub fn mirror_entry(model_id: &str) -> (&str, Option<&'static [&'static str]>) {
    match mirrors::mirror(model_id) {
        Some(Mirror { repo, files, .. }) => (repo, *files),
        None => (model_id, None),
    }
}

/// 下载 `model_id` 的权重到 `dir`（**不**拼 `model_id` 子目录）。
///
/// 模型模块只决定"什么算完整、下完返回哪个文件"，下载本身统一走这里，
/// 不要在模型模块里再写 repo 名或文件清单。
pub async fn download_into(
    model_id: &str,
    dir: &std::path::Path,
    max_retries: u32,
    ctl: DownloadCtl<'_>,
) -> Result<DownloadSummary> {
    let (repo, files) = mirror_entry(model_id);
    let filter = match files {
        Some(files) => DownloadFilter {
            only: Some(files),
            skip_prefixes: &[],
        },
        None => DownloadFilter::ALL,
    };
    download_files_to_with_ctl(repo, dir, max_retries, filter, ctl).await
}

/// 权重落盘目录：`<save_dir>/<model_id>`；`save_dir` 省略时用 `~/.aha` / `$AHA_HOME`。
pub fn model_dir(model_id: &str, save_dir: Option<&str>) -> Result<PathBuf> {
    let base = match save_dir {
        Some(dir) if !dir.is_empty() => dir.to_string(),
        _ => get_default_save_dir().ok_or_else(|| {
            anyhow!("拿不到默认模型目录（home 不在）：Android 上请显式传 save_dir 或设 AHA_HOME")
        })?,
    };
    Ok(PathBuf::from(base).join(model_id))
}

/// 下载一个模型的权重，返回这次真正下了哪些文件（跳过的不算）。
///
/// - 在 [`mirrors::ALL`] 里：只下推理要用的文件，下完校验它们都在（缺一个就报错，
///   避免"半套权重"被当成下载成功）；
/// - 不在表里：把 `model_id` 当成 ModelScope 仓库 id，整仓下载
///   （跟 `aha download -m <任意仓库>` 语义一致）。
///
/// `max_retries` 是每个文件的尝试次数；`ctl` 提供进度回调与取消检查。
pub async fn download_model_files(
    model_id: &str,
    save_dir: Option<&str>,
    max_retries: u32,
    ctl: DownloadCtl<'_>,
) -> Result<DownloadSummary> {
    if model_id.trim().is_empty() {
        bail!("model_id 不能为空");
    }
    let dir = model_dir(model_id, save_dir)?;
    let (repo, files) = mirror_entry(model_id);

    tracing::info!(model_id, repo, dir = %dir.display(), "开始下载模型权重");
    let summary = download_into(model_id, &dir, max_retries, ctl).await?;

    if let Some(files) = files {
        let missing: Vec<&str> = files
            .iter()
            .copied()
            .filter(|file| !dir.join(file).is_file())
            .collect();
        if !missing.is_empty() {
            bail!(
                "{}（{}）下载完仍缺文件：{}",
                model_id,
                repo,
                missing.join(", ")
            );
        }
    }
    tracing::info!(
        model_id,
        repo,
        downloaded = summary.files.len(),
        skipped = summary.skipped,
        "模型权重就绪"
    );
    Ok(summary)
}

/// 这个 id 是否有内置的镜像信息（有 = 只下部分文件，且代码知道它要哪些文件）。
pub fn has_mirror(model_id: &str) -> bool {
    mirrors::mirror(model_id).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_dir_joins_id_under_save_dir() {
        let dir = model_dir("qwen3.5-gguf", Some("/tmp/aha")).unwrap();
        assert_eq!(dir, PathBuf::from("/tmp/aha/qwen3.5-gguf"));
    }

    #[test]
    fn mirrors_cover_the_local_conversions() {
        for id in [
            "qwen3.5-gguf",
            "gpt-sovits-v2pro-zh",
            "OpenBMB/TurboVLA",
            "csukuangfj/kokoro-multi-lang-v1_1",
        ] {
            assert!(has_mirror(id), "{id} 应该在 mirrors::ALL 里");
        }
        assert!(!has_mirror("Qwen/Qwen3-4B"), "普通仓库不该有镜像表项");
    }
}
