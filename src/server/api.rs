use std::pin::pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use aha::models::load_gguf_model;
use aha::models::{GenerateModel, ModelInstance, common::model_mapping::WhichModel, load_model};
use aha::params::chat::ChatCompletionParameters;
use aha::utils::string_to_static_str;
use anyhow::anyhow;
use axum::{
    Json,
    body::{Body, Bytes},
    extract::{ConnectInfo, State, rejection::JsonRejection},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};
use futures_util::{Stream, StreamExt};
use serde::Serialize;
use std::net::SocketAddr;
use tokio::sync::RwLock;

use crate::server::error;
use crate::server::process::cleanup_pid_file;

/// Wrapper to store model type together with the model instance
pub(crate) struct StoredModel {
    pub which_model: WhichModel,
    pub instance: ModelInstance<'static>,
}

// Export MODEL for use in ASR module
pub(crate) static MODEL: OnceLock<Arc<RwLock<StoredModel>>> = OnceLock::new();
static SHUTDOWN_FLAG: OnceLock<Arc<AtomicBool>> = OnceLock::new();
static SERVER_PORT: OnceLock<u16> = OnceLock::new();
static ALLOW_REMOTE_SHUTDOWN: OnceLock<bool> = OnceLock::new();

/// 服务端初始化选项（只对桌宠三件套有意义，其余模型忽略）。
#[derive(Debug, Clone, Default)]
pub struct InitOptions {
    /// Laya 推理设备（cpu / cuda，编译期特征校验）。
    pub device: String,
    /// Laya 置信度门控阈值；<=0 用默认 0.8。
    pub min_confidence: f32,
    /// Laya 单次推理超时（毫秒）；0 用默认 15000。
    pub request_timeout_ms: u64,
    /// sherpa-onnx 的推理线程数；0 = 按 CPU 自动。
    pub threads: i32,
}

/// 默认推理线程数（sherpa-onnx / onnxruntime 的 intra-op 线程）。
///
/// 实测（Ryzen 5 9600X，12 逻辑核 / 6 物理核）：TTS 线程 1/2/4/6/8/12 的 long 文本延迟
/// 是 3619/2962/1973/1574/**1391**/1523ms——8 线程最优，超过 8 后同步开销反噬。
/// 所以默认取 `逻辑核数夹到 4..=8`，不再写死 4。
fn default_threads() -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4)
        .clamp(4, 8)
}

pub fn init(
    model_type: WhichModel,
    path: String,
    gguf: Option<String>,
    mmproj: Option<String>,
    opts: &InitOptions,
) -> anyhow::Result<()> {
    let model = if model_type.is_gguf() {
        if let Some(gguf_path) = gguf {
            let gguf_path = string_to_static_str(gguf_path);
            let mmproj_path = mmproj.map(string_to_static_str);
            load_gguf_model(model_type, None, gguf_path, mmproj_path, None)?
        } else {
            return Err(anyhow!("gguf model need gguf model path"));
        }
    } else if model_type.is_onnx() && !model_type.is_pet_delegated_model() {
        // 桌宠三个模型走 sherpa/sys1，不进 onnx 门禁（SenseVoice 的 id 里就带 "onnx"）
        return Err(anyhow!("onnx comming soon but now not support"));
    } else {
        // Laya 需要设备校验 + 自定义门控/超时，走自己的加载路径。
        #[cfg(feature = "laya")]
        if model_type.is_laya_model() {
            aha::models::laya::check_device(&opts.device)?;
            let model = aha::models::laya::LayaRouteModel::load(
                std::path::Path::new(&path),
                if opts.min_confidence > 0.0 {
                    opts.min_confidence
                } else {
                    aha::models::laya::route::MIN_CONFIDENCE
                },
                opts.request_timeout_ms,
                opts.threads,
            )?;
            MODEL.get_or_init(|| {
                Arc::new(RwLock::new(StoredModel {
                    which_model: model_type,
                    instance: ModelInstance::Laya(model),
                }))
            });
            return Ok(());
        }
        let threads = if opts.threads > 0 {
            opts.threads
        } else {
            default_threads()
        };
        let model_path = string_to_static_str(path);
        load_model(model_type, model_path, None, None, threads)?
    };

    MODEL.get_or_init(|| {
        Arc::new(RwLock::new(StoredModel {
            which_model: model_type,
            instance: model,
        }))
    });
    Ok(())
}

pub fn set_server_port(port: u16, allow_remote_shutdown: bool) {
    SHUTDOWN_FLAG.get_or_init(|| Arc::new(AtomicBool::new(false)));
    SERVER_PORT.get_or_init(|| port);
    ALLOW_REMOTE_SHUTDOWN.get_or_init(|| allow_remote_shutdown);
}

#[allow(unused)]
pub fn get_shutdown_flag() -> Arc<AtomicBool> {
    SHUTDOWN_FLAG
        .get_or_init(|| Arc::new(AtomicBool::new(false)))
        .clone()
}

/// Build a `text/event-stream` response from a stream of already formatted SSE chunks.
fn sse_response<S>(stream: S) -> Response
where
    S: Stream<Item = String> + Send + 'static,
{
    let body = Body::from_stream(stream.map(|chunk| Ok::<_, std::io::Error>(Bytes::from(chunk))));
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(body)
        .expect("failed to build streaming response")
}

fn stored_model() -> Option<Arc<RwLock<StoredModel>>> {
    MODEL.get().cloned()
}

pub(crate) async fn chat(
    payload: Result<Json<ChatCompletionParameters>, JsonRejection>,
) -> Response {
    let req = match payload {
        Ok(Json(req)) => req,
        Err(rejection) => {
            return error::from_json_rejection(rejection).into_response();
        }
    };

    match req.stream {
        Some(false) => {
            let response = {
                let Some(model_ref) = stored_model() else {
                    return error::model_not_loaded().into_response();
                };
                let mut guard = model_ref.write().await;
                guard.instance.generate(req)
            };
            match response {
                Ok(res) => (
                    StatusCode::OK,
                    [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
                    serde_json::to_string(&res).unwrap_or_else(|e| e.to_string()),
                )
                    .into_response(),
                Err(e) => error::internal_error(e.to_string()).into_response(),
            }
        }
        _ => {
            let text_stream = async_stream::stream! {
                let Some(model_ref) = MODEL.get().cloned() else {
                    yield "event: error\ndata: model not init\n\n".to_string();
                    return;
                };
                let mut guard = model_ref.write().await;
                let stream_result = guard.instance.generate_stream(req);
                match stream_result {
                    Ok(stream) => {
                        let mut stream = pin!(stream);
                        while let Some(result) = stream.next().await {
                            match result {
                                Ok(chunk) => {
                                    if let Ok(json_str) = serde_json::to_string(&chunk) {
                                        yield format!("data: {}\n\n", json_str);
                                    }
                                }
                                Err(e) => {
                                    yield format!("data: {{\"error\": \"{}\"}}\n\n", e);
                                    break;
                                }
                            }
                        }
                        yield "data: [DONE]\n\n".to_string();
                    }
                    Err(e) => {
                        yield format!("event: error\ndata: {}\n\n", e.to_string());
                    }
                }
            };
            sse_response(text_stream)
        }
    }
}

pub(crate) async fn remove_background(
    payload: Result<Json<ChatCompletionParameters>, JsonRejection>,
) -> Response {
    let req = match payload {
        Ok(Json(req)) => req,
        Err(rejection) => {
            return error::from_json_rejection(rejection).into_response();
        }
    };
    let response = {
        let Some(model_ref) = stored_model() else {
            return error::model_not_loaded().into_response();
        };
        let mut guard = model_ref.write().await;
        guard.instance.generate(req)
    };
    match response {
        Ok(res) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/json")],
            serde_json::to_string(&res).unwrap(),
        )
            .into_response(),
        Err(e) => error::internal_error(e.to_string()).into_response(),
    }
}

pub(crate) async fn speech(
    payload: Result<Json<ChatCompletionParameters>, JsonRejection>,
) -> Response {
    let req = match payload {
        Ok(Json(req)) => req,
        Err(rejection) => {
            return error::from_json_rejection(rejection).into_response();
        }
    };
    let response = {
        let Some(model_ref) = stored_model() else {
            return error::model_not_loaded().into_response();
        };
        let mut guard = model_ref.write().await;
        guard.instance.generate(req)
    };
    match response {
        Ok(res) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/json")],
            serde_json::to_string(&res).unwrap(),
        )
            .into_response(),
        Err(e) => error::internal_error(e.to_string()).into_response(),
    }
}

// Health check endpoint

#[derive(Serialize)]
pub(crate) struct HealthResponse {
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

pub(crate) async fn health() -> (StatusCode, Json<HealthResponse>) {
    if MODEL.get().is_some() {
        let response = HealthResponse {
            status: "ok".to_string(),
            error: None,
        };
        (StatusCode::OK, Json(response))
    } else {
        let response = HealthResponse {
            status: "unhealthy".to_string(),
            error: Some("model not initialized".to_string()),
        };
        (StatusCode::SERVICE_UNAVAILABLE, Json(response))
    }
}

// Models endpoint (OpenAI-compatible format)

/// OpenAI-compatible model object
#[derive(Serialize)]
struct ModelObject {
    id: String,
    object: String,
    created: Option<i64>,
    owned_by: String,
}

/// OpenAI-compatible models list response
#[derive(Serialize)]
struct ModelsListResponse {
    object: String,
    data: Vec<ModelObject>,
}

pub(crate) async fn models() -> (StatusCode, Json<serde_json::Value>) {
    if let Some(model_ref) = MODEL.get() {
        let guard = model_ref.read().await;
        let which_model = guard.which_model;

        let model_obj = ModelObject {
            id: which_model.as_string(),
            object: "model".to_string(),
            created: None, // We don't track creation time
            owned_by: which_model.model_owner(),
        };
        drop(guard);

        let response = ModelsListResponse {
            object: "list".to_string(),
            data: vec![model_obj],
        };
        (
            StatusCode::OK,
            Json(serde_json::to_value(response).unwrap()),
        )
    } else {
        error::model_not_loaded()
    }
}

// Shutdown endpoint
#[derive(Serialize)]
struct ShutdownResponse {
    message: String,
}

pub(crate) async fn shutdown(
    State(shutdown_flag): State<Arc<AtomicBool>>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
) -> Response {
    // Only loopback clients may shut the service down unless explicitly enabled
    let allow_remote = ALLOW_REMOTE_SHUTDOWN.get().copied().unwrap_or(false);
    let from_loopback = peer.ip().is_loopback();

    // Log the shutdown request
    eprintln!("[SHUTDOWN] Shutdown requested from {peer} (remote_allowed: {allow_remote})");

    if !from_loopback && !allow_remote {
        return error::api_error(
            StatusCode::FORBIDDEN,
            "invalid_request_error",
            "Remote shutdown not allowed. Use --allow-remote-shutdown flag to enable (not recommended).",
            None,
            Some("remote_shutdown_forbidden"),
        )
        .into_response();
    }

    shutdown_flag.store(true, Ordering::SeqCst);

    // Cleanup PID file in a background task
    if let Some(&port) = SERVER_PORT.get() {
        let _ = cleanup_pid_file(port);
    }

    // Schedule shutdown after a short delay to allow response to be sent
    tokio::spawn(async move {
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        std::process::exit(0);
    });

    let response = ShutdownResponse {
        message: "Shutting down...".to_string(),
    };
    (
        StatusCode::OK,
        Json(serde_json::to_value(response).unwrap()),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test health endpoint when model is not initialized
    #[tokio::test]
    async fn test_health_endpoint_uninitialized() {
        let (status, response) = health().await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(response.status, "unhealthy");
        assert_eq!(response.error, Some("model not initialized".to_string()));
    }

    // Test health endpoint when model is initialized
    // Note: This test requires a model to be initialized, which may not be feasible
    // in unit tests without access to model files. This is a placeholder for integration tests.
    //
    // #[tokio::test]
    // async fn test_health_endpoint_initialized() {
    //     // This would require model initialization
    //     // Consider moving to integration tests
    // }

    // Test models endpoint when model is not initialized
    #[tokio::test]
    async fn test_models_endpoint_uninitialized() {
        let (status, response) = models().await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(response.0["error"]["message"], "Model not initialized");
        assert_eq!(response.0["error"]["type"], "service_unavailable");
        assert_eq!(response.0["error"]["code"], "model_not_loaded");
    }

    // Test model type classification
    #[test]
    fn test_get_model_type_llm() {
        assert_eq!(WhichModel::Qwen3_0_6B.model_type(), "llm");
        assert_eq!(WhichModel::MiniCPM4_0_5B.model_type(), "llm");
        assert_eq!(WhichModel::MiniCPM5_1B.model_type(), "llm");
    }

    #[test]
    fn test_get_model_type_vlm() {
        assert_eq!(WhichModel::Qwen3VL2B.model_type(), "vlm");
        assert_eq!(WhichModel::Qwen2_5VL3B.model_type(), "vlm");
        assert_eq!(WhichModel::Qwen2_5VL7B.model_type(), "vlm");
        assert_eq!(WhichModel::Qwen3VL4B.model_type(), "vlm");
        assert_eq!(WhichModel::Qwen3VL8B.model_type(), "vlm");
        assert_eq!(WhichModel::Qwen3VL32B.model_type(), "vlm");
    }

    #[test]
    fn test_get_model_type_ocr() {
        assert_eq!(WhichModel::DeepSeekOCR.model_type(), "ocr");
        assert_eq!(WhichModel::HunyuanOCR.model_type(), "ocr");
        assert_eq!(WhichModel::PaddleOCRVL.model_type(), "ocr");
    }

    #[test]
    fn test_get_model_type_asr() {
        assert_eq!(WhichModel::Qwen3ASR0_6B.model_type(), "asr");
        assert_eq!(WhichModel::Qwen3ASR1_7B.model_type(), "asr");
        assert_eq!(WhichModel::GlmASRNano2512.model_type(), "asr");
        assert_eq!(WhichModel::FunASRNano2512.model_type(), "asr");
    }

    #[test]
    fn test_get_model_type_image() {
        assert_eq!(WhichModel::RMBG2_0.model_type(), "image");
    }

    #[test]
    fn test_get_model_type_tts() {
        assert_eq!(WhichModel::VoxCPM.model_type(), "tts");
        assert_eq!(WhichModel::VoxCPM1_5.model_type(), "tts");
        assert_eq!(WhichModel::VoxCPM2.model_type(), "tts");
    }

    // Remote clients must not be able to stop the service unless explicitly allowed.
    #[tokio::test]
    async fn test_remote_shutdown_forbidden() {
        let flag = Arc::new(AtomicBool::new(false));
        let peer = ConnectInfo(SocketAddr::from(([203, 0, 113, 7], 45678)));
        let response = shutdown(State(flag.clone()), peer).await;
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
        // The shutdown flag must stay untouched on a rejected request.
        assert!(!flag.load(Ordering::SeqCst));

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["code"], "remote_shutdown_forbidden");
    }
}
