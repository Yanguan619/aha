//! Laya 决策模型（sys1 / candle）：判断一句话该走 VLA（身体动作）还是 LLM（对话）。
//!
//! 移植自桌宠仓库 `tools/laya/backend/rust`（laya-route），但模型**原生挂在 aha 里**，
//! 由 aha 的 HTTP 服务通过 `POST /route` 暴露：
//!
//! ```text
//! POST /route  {"text":"走两步"} -> {target,choice,confidence,reason,text}
//! GET  /health                  -> {status:"ok"}
//! ```
//!
//! 设备是 sys1 的**编译期特征**（cpu / cuda 三选一）：用 `--features laya` 编出来是 CPU，
//! `--features laya-cuda` 才是 GPU。`--device` 只是让 exe 校验请求和二进制对不对得上，
//! 对不上直接报错退出——"设置窗里点了 GPU 但桥还在 CPU"是最难查的那类问题。
//!
//! 模型来源按顺序找一个能用的：
//! 1. `--weight-path`（或 `PET_LAYA_CKPT`）——微调版 / 指定目录
//! 2. HF 缓存里已有的 laya checkpoint
//! 3. `--model-id` 从 Hugging Face 下（`HF_ENDPOINT` 可指镜像）

pub mod route;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, anyhow, bail};
use candle_core::DType;
use serde_json::Value;
use sys1::batching::{Batcher, BatcherConfig};
use sys1::models::{
    Architecture, AttentionImplementation, DecisionModel, LAYA_MULTILINGUAL_MODEL_ID,
};
use sys1::schema::{DecisionRequest, DecisionResponse};

use crate::models::laya::route::RouteResult;

/// 解析出来的模型目录 + 来源（hf-cache / download / model-path）。
pub type ResolvedModel = (PathBuf, &'static str);

/// 编进二进制的推理设备。
pub const fn compile_device() -> &'static str {
    if cfg!(feature = "laya-cuda") {
        "cuda"
    } else {
        "cpu"
    }
}

/// 校验请求的设备与二进制编译特征是否一致。
///
/// 设备是**编译期**特征（sys1 的三选一），运行时只能校验，不能切。
pub fn check_device(requested: &str) -> Result<()> {
    let requested = if requested.trim().is_empty() {
        "cpu"
    } else {
        requested.trim()
    };
    match (requested, compile_device()) {
        ("cpu", "cpu") | ("cuda", "cuda") => Ok(()),
        ("cuda", _) => bail!(
            "要 cuda 但这是 cpu 构建：cargo build --release --features laya-cuda（需要 nvcc）"
        ),
        ("cpu", _) => bail!("要 cpu 但这是 cuda 构建：cargo build --release --features laya"),
        other => bail!("不认识的设备 {other:?}（只支持 cpu / cuda）"),
    }
}

/// 常驻的 Laya 引擎：模型只加载一次，请求之间共享 [`Batcher`]（内部有队列 + 动态批处理）。
pub struct LayaRouteModel {
    batcher: Batcher,
    model_id: String,
    min_confidence: f32,
    warmed: AtomicBool,
    request_timeout_ms: u64,
}

impl LayaRouteModel {
    /// 从模型目录加载。`min_confidence` 是 vla 门控阈值（默认 [`route::MIN_CONFIDENCE`]）。
    pub fn load(path: &Path, min_confidence: f32, request_timeout_ms: u64) -> Result<Self> {
        let model = sys1::models::load(
            path,
            Architecture::Laya,
            DType::F32,
            None,
            AttentionImplementation::Eager,
        )
        .with_context(|| format!("加载 Laya 模型失败：{}", path.display()))?;

        let request_timeout_ms = if request_timeout_ms == 0 {
            15_000
        } else {
            request_timeout_ms
        };
        let batcher = Batcher::new(
            Arc::new(model) as Arc<dyn DecisionModel>,
            LAYA_MULTILINGUAL_MODEL_ID.to_owned(),
            BatcherConfig {
                max_batch_size: 4,
                max_batch_questions: 4,
                max_questions_per_request: 1,
                wait: std::time::Duration::ZERO,
                queue_capacity: 32,
                response_timeout: Some(std::time::Duration::from_millis(request_timeout_ms)),
            },
        );

        Ok(Self {
            batcher,
            model_id: LAYA_MULTILINGUAL_MODEL_ID.to_owned(),
            min_confidence: if min_confidence > 0.0 {
                min_confidence
            } else {
                route::MIN_CONFIDENCE
            },
            warmed: AtomicBool::new(false),
            request_timeout_ms,
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn min_confidence(&self) -> f32 {
        self.min_confidence
    }

    pub fn request_timeout_ms(&self) -> u64 {
        self.request_timeout_ms
    }

    /// 首次判定前先空跑一次：sys1 的第一帧推理要挑 kernel，不预热的话
    /// 第一个请求会莫名其妙地慢（和旧 laya-route 的 `Batcher::warmup` 同一个理由）。
    async fn warmup_once(&self) {
        if self.warmed.swap(true, Ordering::SeqCst) {
            return;
        }
        if let Err(error) = self.batcher.warmup().await {
            // 预热失败不影响后续判定（旧 crate 也是只打日志）
            tracing::warn!(error = %error.error, "Laya 预热失败");
        }
    }

    /// 三层判定（和 `route.py` 的 `route_target` 一一对应）。
    pub async fn route(&self, text: &str) -> Result<RouteResult, String> {
        self.warmup_once().await;
        let norm = route::normalize(text);
        if norm.is_empty() {
            return Ok(route::empty(text));
        }
        if route::keyword_hit(&norm) {
            return Ok(route::keyword(text));
        }
        let (choice, confidence) = self.ask(&norm).await?;
        let (target, reason) = route::decide(&choice, confidence, self.min_confidence);
        Ok(RouteResult {
            text: text.to_owned(),
            target,
            choice,
            confidence: Some(confidence),
            reason,
        })
    }

    /// 问模型。只有一条文本、一个问题，所以不需要批处理参数。
    async fn ask(&self, norm: &str) -> Result<(String, f32), String> {
        let request = DecisionRequest {
            model: None,
            state: Value::String(norm.to_owned()),
            questions: route_questions(),
        };
        let response = self
            .batcher
            .predict(request)
            .await
            .map_err(|error| error.error)?;
        parse_answer(&response)
    }
}

/// 和 Python 侧同构的题面：一个问题（route），两个选项（vla / llm）。
pub fn route_questions() -> serde_json::Map<String, Value> {
    let mut questions = serde_json::Map::new();
    questions.insert(
        "route".to_owned(),
        serde_json::json!({
            "type": "choice",
            "instructions": "How should the desktop pet respond to the user's `request`? Pick the execution path.",
            "criteria": {
                "vla": "a direct physical action the pet performs right now: walking, moving, \
                        turning, throwing, jumping, sitting, stopping, or looking at something. \
                        The user is commanding the pet's body, not asking for information.",
                "llm": "a conversation or thinking task: the user is talking to the pet, asking \
                        a question, requesting text, an explanation, or a story."
            }
        }),
    );
    questions
}

/// 从 sys1 的答案里取 `{choice, confidence}`。
///
/// **confidence 的口径必须换算**，否则门控等于换了个阈值：
/// - sys1 的 `confidence` 是**归一化熵** `1 - H/ln(n)`（两个选项时
///   `p=0.96` → 0.756），
/// - Python 的 laya 用 `answer_confidence` = **最大概率**（同一条 → 0.96）。
///
/// `route::MIN_CONFIDENCE=0.8` 是按最大概率定的，所以这里优先从
/// `probabilities` 取 max；只有在 sys1 没给 probabilities 时才退回熵值
/// （那种情况下 0.8 会显得更严，属于安全方向的偏差）。
fn parse_answer(response: &DecisionResponse) -> Result<(String, f32), String> {
    let answer = response
        .answers
        .get("route")
        .ok_or_else(|| "模型没有返回 route 答案".to_owned())?;
    let choice = answer
        .get("choice")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let max_probability = answer
        .get("probabilities")
        .and_then(Value::as_object)
        .and_then(|probabilities| {
            probabilities
                .values()
                .filter_map(Value::as_f64)
                .fold(None, |acc: Option<f64>, value| {
                    Some(acc.map_or(value, |acc| acc.max(value)))
                })
        });
    let confidence = max_probability
        .or_else(|| {
            answer
                .get("confidence")
                .or_else(|| answer.get("answer_confidence"))
                .and_then(Value::as_f64)
        })
        .unwrap_or(0.0) as f32;
    Ok((choice, confidence))
}

// --------------------------------------------------------------------------- //
// 模型来源
// --------------------------------------------------------------------------- //

/// 解析模型目录。`weight_path` 给了就用（必须真的像模型目录），否则先找缓存，
/// 最后才联网下载。
pub async fn resolve(weight_path: Option<&str>, model_id: &str) -> Result<ResolvedModel> {
    if let Some(path) = weight_path.map(Path::new) {
        if looks_like_model(path) {
            return Ok((path.to_path_buf(), "model-path"));
        }
        // 空目录（第一次启动）：当作下载目标
        if !path.exists()
            || std::fs::read_dir(path)
                .map(|mut d| d.next().is_none())
                .unwrap_or(true)
        {
            tracing::info!(dir = %path.display(), "模型目录为空，开始下载 Laya");
            let downloaded = sys1::hub::download(model_id, "main")
                .await
                .context("下载模型失败")?;
            return Ok((downloaded, "download"));
        }
        bail!(
            "{} 不像 laya 模型目录（要 model.safetensors + config.json 或 encoder/config.json）",
            path.display()
        );
    }
    if let Some(path) = discover_cached() {
        return Ok((path, "hf-cache"));
    }
    tracing::info!(
        model_id,
        "本地缓存没有 laya，开始从 Hugging Face 下载（HF_ENDPOINT 可指镜像）"
    );
    let path = sys1::hub::download(model_id, "main")
        .await
        .context("下载 Laya 模型失败")?;
    if !looks_like_model(&path) {
        bail!(
            "下下来了但 {} 不像 laya 模型目录（model.safetensors 缺失？--model-id 选错仓库？）",
            path.display()
        );
    }
    Ok((path, "download"))
}

fn looks_like_model(path: &Path) -> bool {
    path.join("model.safetensors").is_file()
        && (path.join("config.json").is_file() || path.join("encoder/config.json").is_file())
}

/// 找 Python 版留在 HF 缓存里的 checkpoint。
///
/// 两种布局都认：
/// - `models--convaiinnovations--laya/snapshots/<rev>/multilingual/`
///   （Python `laya` 库把 multilingual 放在子目录）
/// - `models--convaiinnovations--laya-multilingual/snapshots/<rev>/`
fn discover_cached() -> Option<PathBuf> {
    let hub = hf_hub_dir()?;
    let candidates = [
        ("models--convaiinnovations--laya", Some("multilingual")),
        ("models--convaiinnovations--laya-multilingual", None),
        ("models--convaiinnovations--laya", Some("")),
    ];
    for (repo, subdir) in candidates {
        let snapshots = hub.join(repo).join("snapshots");
        let Ok(entries) = std::fs::read_dir(&snapshots) else {
            continue;
        };
        let mut dirs: Vec<PathBuf> = entries
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.is_dir())
            .collect();
        // 新的快照优先（同一份权重的不同 revision 内容一样，取最新的最省事）
        dirs.sort_by_key(|path| {
            std::fs::metadata(path)
                .and_then(|meta| meta.modified())
                .ok()
        });
        for dir in dirs.into_iter().rev() {
            let path = match subdir {
                Some(sub) if !sub.is_empty() => dir.join(sub),
                _ => dir,
            };
            if looks_like_model(&path) {
                return Some(path);
            }
        }
    }
    None
}

fn hf_hub_dir() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("HF_HUB_CACHE") {
        return Some(PathBuf::from(path));
    }
    if let Some(path) = std::env::var_os("HF_HOME") {
        return Some(PathBuf::from(path).join("hub"));
    }
    let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"))?;
    Some(
        PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("hub"),
    )
}

/// 验收语料跑一遍（给 `aha probe --model <laya>` / 手工自检用）。
pub async fn selftest(model: &LayaRouteModel) -> Result<()> {
    let started = std::time::Instant::now();
    let mut failures = 0usize;
    let mut raw_hits = 0usize;

    println!("\n原始判定明细（raw = 模型/关键词的初判，target = 门控后的最终值）");
    println!("{}", "-".repeat(92));
    for (want, text) in route::CORPUS {
        let result = model
            .route(text)
            .await
            .map_err(|error| anyhow!("{text:?} 判定失败：{error}（模型路径没跑通？）"))?;
        let raw_hit = result.choice == *want;
        let gated_hit = result.target == *want;
        if raw_hit {
            raw_hits += 1;
        }
        if !gated_hit {
            failures += 1;
        }
        let confidence = match result.confidence {
            Some(value) => format!("{value:.3}"),
            None => "-".to_owned(),
        };
        println!(
            "{} want={:3} raw={:3} conf={:5} target={:3} reason={:14} {text:?}",
            if gated_hit { "ok " } else { "XX " },
            want,
            result.choice,
            confidence,
            result.target,
            result.reason,
        );
    }
    println!("{}", "-".repeat(92));
    for text in route::AMBIGUOUS {
        let result = model.route(text).await.map_err(anyhow::Error::msg)?;
        if result.target != "llm" {
            failures += 1;
            println!("XX  模糊输入走了 vla：{text:?} -> {result:?}");
        }
    }
    let total = route::CORPUS.len();
    println!("raw 命中  : {raw_hits}/{total}");
    println!(
        "gated 命中: {}/{total}  (gate={}, model={}, device={})",
        total - failures,
        model.min_confidence(),
        model.model_id(),
        compile_device()
    );
    println!("耗时 {:.1}s", started.elapsed().as_secs_f32());

    if failures > 0 {
        bail!("{failures} 条判定不符合标注");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sys1::schema::Usage;

    fn response(answer: Value) -> DecisionResponse {
        let mut answers = serde_json::Map::new();
        answers.insert("route".to_owned(), answer);
        DecisionResponse {
            model: "test".to_owned(),
            answers,
            usage: Usage {
                input_tokens: 0,
                output_tokens: 0,
            },
        }
    }

    #[test]
    fn parses_sys1_answer_as_max_probability() {
        // sys1 的 confidence 是熵（0.756），门控要的是最大概率（0.96）
        let got = parse_answer(&response(serde_json::json!({
            "type": "choice",
            "choice": "vla",
            "probabilities": {"vla": 0.96, "llm": 0.04},
            "confidence": 0.756
        })))
        .unwrap();
        assert_eq!(got.0, "vla");
        assert!((got.1 - 0.96).abs() < 1e-6, "got {}", got.1);
    }

    #[test]
    fn falls_back_to_entropy_when_probabilities_missing() {
        let got = parse_answer(&response(serde_json::json!({
            "type": "choice", "choice": "llm", "confidence": 0.42
        })))
        .unwrap();
        assert_eq!(got, ("llm".to_owned(), 0.42));
    }

    #[test]
    fn parses_python_style_confidence() {
        let got = parse_answer(&response(serde_json::json!({
            "type": "choice", "choice": "llm", "answer_confidence": 0.42
        })))
        .unwrap();
        assert_eq!(got, ("llm".to_owned(), 0.42));
    }

    #[test]
    fn missing_answer_is_an_error() {
        let mut answers = serde_json::Map::new();
        answers.insert("other".to_owned(), serde_json::json!({}));
        let err = parse_answer(&DecisionResponse {
            model: "test".to_owned(),
            answers,
            usage: Usage {
                input_tokens: 0,
                output_tokens: 0,
            },
        })
        .unwrap_err();
        assert!(err.contains("route"));
    }

    #[test]
    fn questions_keep_both_criteria() {
        let questions = route_questions();
        let criteria = questions["route"]["criteria"].as_object().unwrap();
        assert!(criteria.contains_key("vla") && criteria.contains_key("llm"));
    }
}
