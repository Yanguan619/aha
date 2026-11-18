//! 下载机制：列仓库文件、按 `Range` 续传、进度回调、重试。
//!
//! 这一层只跟 HTTP / 文件打交道，**不认识模型**：仓库 id 与文件清单来自
//! [`crate::models::mirrors`]，策略（`ensure_*` / `download_into`）在同目录的 `mod.rs`。
//!
//! 模型模块只允许调 `ensure_*`；要拼 `DownloadFilter` 说明该能力还没抽出来，
//! 应该加在 `mod.rs` 里而不是绕到这里。

use std::fs;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use futures_util::StreamExt;
use tokio::time::sleep;

pub async fn download_model(
    model_id: &str,
    save_dir: &str,
    max_retries: u32,
) -> anyhow::Result<()> {
    let dir = PathBuf::from(save_dir).join(model_id);
    download_files_to(model_id, &dir, max_retries, DownloadFilter::ALL)
        .await
        .map(|_| ())
}

/// 下载过滤：`only` 非空时只下这些相对路径——桌宠的 sherpa/laya 只要几个权重，
/// 而镜像仓库里通常还放着一份 fp32 的重复模型（SenseVoice 的 `model.onnx` 就有
/// 900MB+），没必要下。`skip_prefixes` 按路径前缀整段跳过。
#[derive(Clone, Copy)]
pub struct DownloadFilter<'a> {
    /// 只要这些文件（相对仓库根的路径）；`None` = 仓库里全部文件。
    pub only: Option<&'a [&'a str]>,
    /// 跳过这些路径前缀。
    pub skip_prefixes: &'a [&'a str],
}

impl DownloadFilter<'static> {
    /// 整个仓库都下（`aha download` 的默认行为）。
    pub const ALL: DownloadFilter<'static> = DownloadFilter {
        only: None,
        skip_prefixes: &[],
    };
}

/// 一次下载的进度快照。`done` / `total` 是整次任务，`file_done` / `file_total` 是当前文件。
#[derive(Clone, Copy)]
pub struct DownloadProgress<'a> {
    pub path: &'a str,
    pub file_done: u64,
    pub file_total: u64,
    pub done: u64,
    pub total: u64,
    pub files_done: usize,
    pub files_total: usize,
}

/// 下载控制：进度回调与取消检查都可以不给。
///
/// 回调在下载线程里同步触发（约 100ms 节流一次），别在里面做耗时操作；
/// FFI 那层把它转成 C 函数指针后，调用方要保证 `user` 在这次下载期间一直有效。
#[derive(Clone, Copy, Default)]
pub struct DownloadCtl<'a> {
    pub progress: Option<&'a dyn Fn(&DownloadProgress<'_>)>,
    pub cancel: Option<&'a dyn Fn() -> bool>,
}

impl DownloadCtl<'_> {
    fn cancelled(&self) -> bool {
        self.cancel.is_some_and(|cancel| cancel())
    }
}

/// 下载结果：真正写盘的文件（相对仓库根的路径, 字节数）。
#[derive(Default, Debug)]
pub struct DownloadSummary {
    pub files: Vec<(String, u64)>,
    pub skipped: usize,
}

struct ProgressTracker<'a> {
    ctl: DownloadCtl<'a>,
    total: u64,
    done: u64,
    files_done: usize,
    files_total: usize,
    last: Instant,
}

impl<'a> ProgressTracker<'a> {
    fn new(ctl: DownloadCtl<'a>, total: u64, files_total: usize) -> Self {
        Self {
            ctl,
            total,
            done: 0,
            files_done: 0,
            files_total,
            last: Instant::now(),
        }
    }

    fn check_cancel(&self) -> anyhow::Result<()> {
        if self.ctl.cancelled() {
            anyhow::bail!("下载已取消");
        }
        Ok(())
    }

    fn skip(&mut self, size: u64) {
        self.done += size;
        self.files_done += 1;
    }

    /// 断点续传：把上次已经落盘的字节补进总进度（不动 `files_done`，那个文件还没下完）。
    fn resume(&mut self, bytes: u64) {
        self.done += bytes;
    }

    /// 当前文件又收到 `n` 字节。
    fn advance(&mut self, path: &str, n: u64, file_done: u64, file_total: u64) {
        self.done += n;
        self.emit(path, file_done, file_total, false);
    }

    fn finish_file(&mut self, path: &str, size: u64) {
        self.files_done += 1;
        self.emit(path, size, size, true);
    }

    fn emit(&mut self, path: &str, file_done: u64, file_total: u64, force: bool) {
        if !force && self.last.elapsed() < Duration::from_millis(100) {
            return;
        }
        self.last = Instant::now();
        if let Some(cb) = self.ctl.progress {
            cb(&DownloadProgress {
                path,
                file_done,
                file_total,
                done: self.done,
                total: self.total,
                files_done: self.files_done,
                files_total: self.files_total,
            });
        }
    }
}

/// 把 `repo_id` 指定的 ModelScope 仓库文件下到 `dir`，失败按 `max_retries` 重试。
///
/// 已存在且大小一致的文件跳过；剩下的从
/// `{MS_RESOLVE}/{repo_id}/resolve/master/{path}` 流式写盘。**单文件支持断点续传**：
/// `<file>.part` 存在就按它的长度发 `Range`，服务端不认（200）或对不上（416）时退回整份重下。
pub async fn download_files_to(
    repo_id: &str,
    dir: &Path,
    max_retries: u32,
    filter: DownloadFilter<'_>,
) -> anyhow::Result<DownloadSummary> {
    download_files_to_with_ctl(repo_id, dir, max_retries, filter, DownloadCtl::default()).await
}

/// 同上，但可以挂进度回调和取消检查（FFI 层用这个）。
pub async fn download_files_to_with_ctl(
    repo_id: &str,
    dir: &Path,
    max_retries: u32,
    filter: DownloadFilter<'_>,
    ctl: DownloadCtl<'_>,
) -> anyhow::Result<DownloadSummary> {
    let max_retries = max_retries.max(1);
    let mut attempts = 0u32;
    loop {
        attempts += 1;
        println!("Attempting to download {repo_id} (attempt {attempts}/{max_retries})");

        match download_repo(repo_id, dir, filter, ctl).await {
            Ok(summary) => {
                println!("Model downloaded successfully");
                return Ok(summary);
            }
            Err(e) => {
                if attempts >= max_retries || ctl.cancelled() {
                    return Err(anyhow::anyhow!(
                        "Failed to download {repo_id} after {attempts} attempts. Last error: {e}"
                    ));
                }

                println!("Download failed (attempt {attempts}): {e}. Retrying in 2 seconds...");
                sleep(Duration::from_secs(2)).await;
            }
        }
    }
}

/// ModelScope 上模型的文件列表；字段名是接口原始的大写驼峰。
#[derive(serde::Deserialize)]
struct RepoFilesResponse {
    #[serde(rename = "Success")]
    success: bool,
    #[serde(rename = "Message")]
    message: String,
    #[serde(rename = "Data")]
    data: Option<RepoFilesData>,
}

#[derive(serde::Deserialize)]
struct RepoFilesData {
    #[serde(rename = "Files")]
    files: Vec<RepoFile>,
}

#[derive(serde::Deserialize)]
struct RepoFile {
    #[serde(rename = "Path")]
    path: String,
    #[serde(rename = "Size")]
    size: u64,
    #[serde(rename = "Type")]
    kind: String,
}

const MS_API: &str = "https://modelscope.cn/api/v1/models";
const MS_RESOLVE: &str = "https://modelscope.cn/models";
const MS_UA: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
                     (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36";
/// 小于这个字节数的文件不核实际大小（tokens.txt 这类只有几百字节）。
const TINY_BYTES: u64 = 4096;

/// 把一个 ModelScope 仓库的文件下到 `dir`（已存在且大小一致的跳过）。
///
/// 自己写而不用 `modelscope` crate：它的 reqwest 走平台证书校验，Android 上直接 panic，
/// 见 [`http`]。顺手也去掉了那个依赖。
async fn download_repo(
    repo_id: &str,
    dir: &Path,
    filter: DownloadFilter<'_>,
    ctl: DownloadCtl<'_>,
) -> Result<DownloadSummary> {
    if ctl.cancelled() {
        anyhow::bail!("下载已取消");
    }
    fs::create_dir_all(dir)?;
    println!("\nDownloading model {repo_id} to: {}\n", dir.display());

    let client = crate::utils::http::client()?;
    let resp = client
        .get(format!("{MS_API}/{repo_id}/repo/files?Recursive=true"))
        .header("User-Agent", MS_UA)
        .send()
        .await?;
    if !resp.status().is_success() {
        anyhow::bail!(
            "Failed to list files of {repo_id}: HTTP {}（模型 id 写错了，或者需要登录）",
            resp.status()
        );
    }
    let body: RepoFilesResponse = resp.json().await?;
    if !body.success {
        anyhow::bail!("Failed to list files of {repo_id}: {}", body.message);
    }
    let files = body
        .data
        .ok_or_else(|| anyhow!("Failed to list files of {repo_id}: empty response"))?
        .files;
    let blobs: Vec<RepoFile> = files
        .into_iter()
        .filter(|f| f.kind == "blob")
        .filter(|f| match filter.only {
            Some(only) => only.contains(&f.path.as_str()),
            None => true,
        })
        .filter(|f| {
            !filter
                .skip_prefixes
                .iter()
                .any(|prefix| f.path.starts_with(prefix))
        })
        .collect();
    if blobs.is_empty() {
        match filter.only {
            Some(only) => anyhow::bail!(
                "{repo_id} 里没有这些文件：{}（仓库的文件列表变了？）",
                only.join(", ")
            ),
            None => anyhow::bail!("{repo_id} 的文件列表是空的"),
        }
    }

    let mut summary = DownloadSummary::default();
    let total: u64 = blobs.iter().map(|file| file.size).sum();
    let mut tracker = ProgressTracker::new(ctl, total, blobs.len());
    for file in &blobs {
        tracker.check_cancel()?;
        let dst = dir.join(&file.path);
        if fs::metadata(&dst)
            .map(|m| m.len() == file.size)
            .unwrap_or(false)
        {
            summary.skipped += 1;
            tracker.skip(file.size);
            continue;
        }
        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent)?;
        }
        download_file(&client, repo_id, file, &dst, &mut tracker).await?;
        println!("  {} ({:.1} MB)", file.path, file.size as f64 / 1_048_576.0);
        tracker.finish_file(&file.path, file.size);
        summary.files.push((file.path.clone(), file.size));
    }
    Ok(summary)
}

/// 一个带可选 `Range` 的 GET。`start == 0` 表示整份下载。
///
/// 单独抽出来是因为 416 的兜底要再发一次不带 `Range` 的请求，两处必须用同一套 header。
async fn get_with_range(
    client: &reqwest::Client,
    url: &str,
    start: u64,
) -> Result<reqwest::Response> {
    let mut request = client.get(url).header("User-Agent", MS_UA);
    if start > 0 {
        request = request.header("Range", format!("bytes={start}-"));
    }
    Ok(request.send().await?)
}

async fn download_file(
    client: &reqwest::Client,
    repo_id: &str,
    file: &RepoFile,
    dst: &Path,
    tracker: &mut ProgressTracker<'_>,
) -> Result<()> {
    // 断点续传：`<file>.part` 就是上一次写了一半的文件，按它的大小发 `Range` 接着写。
    // 服务端不认 `Range`（回 200）或 `.part` 与远端对不上（回 416）时退回整份重下。
    let part = dst.with_file_name(format!(
        "{}.part",
        dst.file_name().unwrap_or_default().to_string_lossy()
    ));
    let mut have = fs::metadata(&part).map(|meta| meta.len()).unwrap_or(0);
    // 残留的 `.part` 比声明大小还大 = 上游换过版本，接着写只会得到坏文件，扔掉重来。
    if have > file.size {
        println!(
            "  {} 的残留 .part（{have} B）超过声明大小（{} B），丢弃重下",
            file.path, file.size
        );
        let _ = fs::remove_file(&part);
        have = 0;
    }

    let url = format!("{MS_RESOLVE}/{repo_id}/resolve/master/{}", file.path);
    let mut resp = get_with_range(client, &url, have).await?;
    if have > 0 && resp.status() == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
        println!(
            "  {} 的 .part 无法续传（HTTP 416），丢弃后整份重下",
            file.path
        );
        let _ = fs::remove_file(&part);
        have = 0;
        resp = get_with_range(client, &url, 0).await?;
    }
    if !resp.status().is_success() {
        anyhow::bail!("Failed to download {}: HTTP {}", file.path, resp.status());
    }
    let append = have > 0 && resp.status() == reqwest::StatusCode::PARTIAL_CONTENT;
    if have > 0 && !append {
        println!(
            "  {} 的服务器忽略了 Range（HTTP {}），从头下载",
            file.path,
            resp.status()
        );
    }

    // 先写 `<file>.part`，校验大小后再改名：进程中途被杀也不会留下"看起来完整"的半截文件
    // （`ensure_files` 这类调用只看文件在不在，半截文件会被当成已下完）。
    let mut out = if append {
        // 续上的那部分也算进度，否则进度条会从 0 开始骗人。
        tracker.resume(have);
        std::fs::OpenOptions::new().append(true).open(&part)?
    } else {
        File::create(&part)?
    };
    let mut written = if append { have } else { 0u64 };
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        // 取消/退出时立即中断，`.part` 留着，下次从这个位置继续
        tracker.check_cancel()?;
        let chunk = chunk?;
        written += chunk.len() as u64;
        out.write_all(&chunk)?;
        tracker.advance(&file.path, chunk.len() as u64, written, file.size);
    }
    out.flush()?;
    drop(out);
    // 大小对不上就是没下完（CDN 偶尔截断）：留着 .part，下次重下
    if file.size >= TINY_BYTES && written != file.size {
        anyhow::bail!(
            "下载不完整：{} 得到 {written} 字节，期望 {}",
            file.path,
            file.size
        );
    }
    fs::rename(&part, dst)?;
    Ok(())
}
