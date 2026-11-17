use crate::server::api::set_server_port;
use crate::server::process::{cleanup_pid_file, create_pid_file};
use axum::{
    Router,
    extract::DefaultBodyLimit,
    middleware,
    routing::{get, post},
};
use std::net::IpAddr;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tower_http::trace::TraceLayer;

// ASR (Automatic Speech Recognition) API module
pub(crate) mod api;
pub(crate) mod asr;
#[cfg(feature = "laya")]
pub(crate) mod decisions;
pub(crate) mod embedding;
pub(crate) mod error;
pub(crate) mod metrics;
pub(crate) mod process;
/// `/realtime`：OpenAI Realtime 形状的 WebSocket 实时语音端点（ASR → LLM → TTS）。
pub(crate) mod realtime;
pub(crate) mod reranker;
/// TTS：OpenAI 形状的 `/audio/speech` + 桌宠两个别名端点共用的实现。
#[cfg(any(feature = "sherpa-tts", feature = "gpt-sovits"))]
pub(crate) mod tts;
#[cfg(feature = "vla")]
pub(crate) mod vla;
pub(crate) mod warmup;

/// Default request body limit (5 MiB), same as the previous rocket configuration.
const DEFAULT_BODY_LIMIT: usize = 5 * 1024 * 1024;
/// Multipart upload limit for audio transcription endpoints (100 MiB).
const MULTIPART_BODY_LIMIT: usize = 100 * 1024 * 1024;

pub(crate) async fn start_http_server(
    address: String,
    port: u16,
    allow_remote_shutdown: bool,
    parent_pid: u32,
) -> anyhow::Result<()> {
    // Set server port for shutdown endpoint
    set_server_port(port, allow_remote_shutdown);
    metrics::set_port(port);

    // 桌宠委托模式：父进程（pet.exe）一死就跟着退出，别留孤儿占着端口。
    // 和旧 asr-route / tts-route / laya-route 的 --parent-pid 看门狗同一个约定。
    if parent_pid > 0 {
        process::watch_parent(parent_pid);
    }

    // Create PID file for service tracking
    let pid = std::process::id();
    create_pid_file(pid, port)?;

    // Set up shutdown flag
    let shutdown_flag = Arc::new(AtomicBool::new(false));
    let shutdown_flag_clone = shutdown_flag.clone();

    // Configure Ctrl+C handler for graceful shutdown
    let shutdown_handler = tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        println!("Received shutdown signal, gracefully shutting down...");
        shutdown_flag_clone.store(true, Ordering::SeqCst);
        // Give time for existing requests to complete
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        // Cleanup PID file
        let _ = cleanup_pid_file(port);
        std::process::exit(0);
    });

    let app = build_router(shutdown_flag);

    let ip = IpAddr::from_str(&address)?;
    let listener = tokio::net::TcpListener::bind((ip, port)).await?;

    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .await?;

    // Cleanup PID file when server exits
    cleanup_pid_file(port)?;
    shutdown_handler.abort();

    Ok(())
}

/// Build the axum router serving every aha HTTP endpoint.
fn build_router(shutdown_flag: Arc<AtomicBool>) -> Router {
    let app = Router::new()
        // /v1/chat/completions and /chat/completions
        .route("/v1/chat/completions", post(api::chat))
        .route("/chat/completions", post(api::chat))
        // /images/remove_background
        .route("/images/remove_background", post(api::remove_background))
        // /audio/speech
        .route("/audio/speech", post(api::speech))
        // /embeddings and /v1/embeddings (OpenAI-compatible embeddings endpoint)
        .route("/embeddings", post(embedding::embeddings))
        .route("/v1/embeddings", post(embedding::embeddings))
        // /rerank and /v1/rerank (OpenAI-compatible rerank endpoint)
        .route("/rerank", post(reranker::rerank))
        .route("/v1/rerank", post(reranker::rerank))
        // Health check and model info endpoints
        .route("/health", get(api::health))
        .route("/models", get(api::models))
        // 运行时指标（JSON）+ 检测面板（纯前端，轮询 /metrics）
        .route("/metrics", get(metrics::metrics_handler))
        .route("/", get(metrics::panel))
        // OpenAI-compatible model listing endpoint: /v1/models
        .route("/v1/models", get(api::models))
        // Shutdown endpoint
        .route("/shutdown", post(api::shutdown))
        // /realtime and /v1/realtime (WebSocket：ASR → LLM → TTS 对流式对话)
        .route("/realtime", get(realtime::handler))
        .route("/v1/realtime", get(realtime::handler))
        // JSON endpoints share the default request body limit
        .layer(DefaultBodyLimit::max(DEFAULT_BODY_LIMIT))
        // /audio/transcriptions and /v1/audio/transcriptions (ASR transcription endpoint)
        .route(
            "/audio/transcriptions",
            post(asr::transcriptions).layer(DefaultBodyLimit::max(MULTIPART_BODY_LIMIT)),
        )
        .route(
            "/v1/audio/transcriptions",
            post(asr::transcriptions).layer(DefaultBodyLimit::max(MULTIPART_BODY_LIMIT)),
        );

    // 剩下的桌宠风味只有一条：Laya 判定。它是**标准 Jev 形状**
    // （{state,questions} → {model,answers,usage}，和 Ollama /v1/systemone 同一套），
    // 题面由调用方在请求里传，端点本身不携带桌宠策略。
    //
    // 旧的 /v1/asr（裸 PCM）、/v1/tts（WAV）、/v1/tts/stream（PCM 流）三条别名已删除：
    // 通用面分别是 `/v1/audio/transcriptions`（非 multipart 分支）和 `/audio/speech`
    // （`response_format: wav|pcm`）。
    #[cfg(feature = "laya")]
    let app = app.route("/v1/decisions", post(decisions::handle_route));
    #[cfg(feature = "vla")]
    let app = app.route("/v1/actions", post(vla::handle_actions));

    app
        // Unknown routes and wrong methods also answer with the OpenAI error envelope
        .fallback(error::fallback_not_found)
        .method_not_allowed_fallback(error::fallback_method_not_allowed)
        // Rewrite any framework-generated error (413/415/...) into the same envelope
        .layer(middleware::from_fn(error::normalize_error_response))
        .layer(TraceLayer::new_for_http())
        // 最外层：统计到的状态码/耗时就是客户端最终看到的
        .layer(middleware::from_fn(metrics::track_metrics))
        .with_state(shutdown_flag)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Start a router on an ephemeral port and return its base URL.
    async fn spawn_server() -> String {
        let app = build_router(Arc::new(AtomicBool::new(false)));
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(
                listener,
                app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
            )
            .await
            .unwrap();
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn test_health_route_without_model() {
        let base = spawn_server().await;
        let response = reqwest::get(format!("{base}/health")).await.unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);
        let body: serde_json::Value =
            serde_json::from_str(&response.text().await.unwrap()).unwrap();
        assert_eq!(body["status"], "unhealthy");
        assert_eq!(body["error"], "model not initialized");
    }

    #[tokio::test]
    async fn test_models_route_without_model() {
        let base = spawn_server().await;
        for path in ["/models", "/v1/models"] {
            let response = reqwest::get(format!("{base}{path}")).await.unwrap();
            assert_eq!(
                response.status(),
                reqwest::StatusCode::SERVICE_UNAVAILABLE,
                "unexpected status for {path}"
            );
            let body: serde_json::Value =
                serde_json::from_str(&response.text().await.unwrap()).unwrap();
            assert_eq!(body["error"]["message"], "Model not initialized");
            assert_eq!(body["error"]["type"], "service_unavailable");
        }
    }

    #[tokio::test]
    async fn test_chat_route_reachable_without_model() {
        let base = spawn_server().await;
        let response = reqwest::Client::new()
            .post(format!("{base}/v1/chat/completions"))
            .json(&serde_json::json!({
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": false
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);
        let body: serde_json::Value =
            serde_json::from_str(&response.text().await.unwrap()).unwrap();
        assert_eq!(body["error"]["message"], "Model not initialized");
    }

    #[tokio::test]
    async fn test_invalid_json_returns_openai_error() {
        let base = spawn_server().await;
        let response = reqwest::Client::new()
            .post(format!("{base}/v1/chat/completions"))
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body("{ not valid json ")
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
        let body: serde_json::Value =
            serde_json::from_str(&response.text().await.unwrap()).unwrap();
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], "invalid_body");
    }

    #[tokio::test]
    async fn test_unknown_route_returns_404() {
        let base = spawn_server().await;
        let response = reqwest::get(format!("{base}/not-exist")).await.unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::NOT_FOUND);
        let body: serde_json::Value =
            serde_json::from_str(&response.text().await.unwrap()).unwrap();
        assert_eq!(body["error"]["code"], "unknown_endpoint");
        assert_eq!(body["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn test_wrong_method_returns_openai_error() {
        let base = spawn_server().await;
        let response = reqwest::Client::new()
            .get(format!("{base}/v1/chat/completions"))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::METHOD_NOT_ALLOWED);
        let body: serde_json::Value =
            serde_json::from_str(&response.text().await.unwrap()).unwrap();
        assert_eq!(body["error"]["code"], "method_not_allowed");
    }

    #[tokio::test]
    async fn test_oversized_body_returns_openai_error() {
        let base = spawn_server().await;
        // The default JSON body limit is 5 MiB.
        let oversized = "x".repeat(DEFAULT_BODY_LIMIT + 1024);
        let response = reqwest::Client::new()
            .post(format!("{base}/v1/chat/completions"))
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(oversized)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::PAYLOAD_TOO_LARGE);
        let body: serde_json::Value =
            serde_json::from_str(&response.text().await.unwrap()).unwrap();
        assert_eq!(body["error"]["message"], "Request body is too large");
    }

    #[tokio::test]
    async fn test_metrics_route() {
        let base = spawn_server().await;
        let response = reqwest::get(format!("{base}/metrics")).await.unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let body: serde_json::Value =
            serde_json::from_str(&response.text().await.unwrap()).unwrap();
        assert!(body["server"]["pid"].is_number());
        assert!(body["routes"].is_array());
    }

    #[tokio::test]
    async fn test_panel_route() {
        let base = spawn_server().await;
        let response = reqwest::get(format!("{base}/")).await.unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_string();
        assert!(
            content_type.contains("text/html"),
            "unexpected content-type: {content_type}"
        );
        assert!(response.text().await.unwrap().contains("<html"));
    }

    #[tokio::test]
    async fn test_transcription_multipart_reaches_model() {
        let base = spawn_server().await;
        let boundary = "----ahaTestBoundary";
        let body = format!(
            "--{boundary}\r\n\
             Content-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\n\
             Content-Type: audio/wav\r\n\r\n\
             fake-audio-bytes\r\n\
             --{boundary}\r\n\
             Content-Disposition: form-data; name=\"language\"\r\n\r\n\
             zh\r\n\
             --{boundary}--\r\n"
        );
        let response = reqwest::Client::new()
            .post(format!("{base}/v1/audio/transcriptions"))
            .header(
                reqwest::header::CONTENT_TYPE,
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(body)
            .send()
            .await
            .unwrap();
        // Multipart parsed fine and hit the handler; the model simply is not loaded.
        assert_eq!(response.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);
        let body: serde_json::Value =
            serde_json::from_str(&response.text().await.unwrap()).unwrap();
        assert_eq!(body["error"]["message"], "Model not initialized");
    }

    /// `/realtime` 的真实握手与事件往返。这里不加载任何模型，只验证协议层：
    /// 握手 → session.created → session.update 合并 → 音频提交 → 模型缺失时
    /// 报错但连接不断。
    #[tokio::test]
    async fn test_realtime_websocket_protocol() {
        use futures_util::{SinkExt, StreamExt};
        use tokio_tungstenite::tungstenite::Message;

        type Socket = tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >;

        /// 读下一条事件；JSON 解析成 `Value`。
        async fn next_event(socket: &mut Socket) -> serde_json::Value {
            let frame = tokio::time::timeout(std::time::Duration::from_secs(10), socket.next())
                .await
                .expect("等事件超时")
                .expect("连接被关闭")
                .expect("读帧失败");
            let Message::Text(text) = frame else {
                panic!("期望文本帧，收到 {frame:?}");
            };
            serde_json::from_str(text.as_str()).expect("事件不是 JSON")
        }

        async fn send(socket: &mut Socket, value: serde_json::Value) {
            socket
                .send(Message::Text(value.to_string().into()))
                .await
                .unwrap();
        }

        let base = spawn_server().await;
        let url = base.replace("http://", "ws://") + "/realtime?voice=af_heart";
        let (mut socket, response) = tokio_tungstenite::connect_async(url).await.unwrap();
        assert_eq!(response.status().as_u16(), 101, "WebSocket 握手失败");

        let created = next_event(&mut socket).await;
        assert_eq!(created["type"], "session.created");
        assert!(
            created["session"]["id"]
                .as_str()
                .unwrap()
                .starts_with("sess_")
        );
        // 查询参数里的音色生效
        assert_eq!(created["session"]["voice"], "af_heart");
        assert_eq!(created["session"]["output_audio_sample_rate"], 24_000);
        assert_eq!(created["session"]["input_audio_format"], "pcm16");

        // session.update 合并字段，并回一条 session.updated
        send(
            &mut socket,
            serde_json::json!({
                "type": "session.update",
                "session": {
                    "modalities": ["audio", "text"],
                    "instructions": "你是一个桌宠",
                    "turn_detection": { "type": "server_vad", "silence_duration_ms": 800 },
                }
            }),
        )
        .await;
        let updated = next_event(&mut socket).await;
        assert_eq!(updated["type"], "session.updated");
        assert_eq!(updated["session"]["instructions"], "你是一个桌宠");
        assert_eq!(updated["session"]["modalities"][1], "text");
        assert_eq!(
            updated["session"]["turn_detection"]["silence_duration_ms"],
            800
        );

        // 不认识的类型 → error 事件，连接不断
        send(&mut socket, serde_json::json!({ "type": "nope" })).await;
        let error = next_event(&mut socket).await;
        assert_eq!(error["type"], "error");
        assert_eq!(error["error"]["type"], "invalid_request_error");

        // 二进制帧 = append 音频；手动 commit 时没有 ASR 模型，应报 transcription_failed，
        // 但 input_audio_buffer.committed 照发，连接照旧可用。
        socket
            .send(Message::Binary(vec![0u8; 4_800 * 2].into()))
            .await
            .unwrap();
        send(
            &mut socket,
            serde_json::json!({ "type": "input_audio_buffer.commit" }),
        )
        .await;
        let mut saw_transcription_error = false;
        let mut saw_committed = false;
        // committed 先到、识别结果（这里是错误）后到，两个都收齐再断言。
        for _ in 0..8 {
            let event = next_event(&mut socket).await;
            if event["type"] == "error" {
                saw_transcription_error |= event["error"]["code"] == "transcription_failed";
            }
            if event["type"] == "input_audio_buffer.committed" {
                saw_committed = true;
            }
            if saw_committed && saw_transcription_error {
                break;
            }
        }
        assert!(saw_committed, "没有收到 input_audio_buffer.committed");
        assert!(
            saw_transcription_error,
            "没有收到 transcription_failed 错误"
        );

        // 文本 item + response.create：没有 LLM 时必须 model_not_loaded + response.done
        send(
            &mut socket,
            serde_json::json!({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{ "type": "input_text", "text": "你好" }],
                }
            }),
        )
        .await;
        let item = next_event(&mut socket).await;
        assert_eq!(item["type"], "conversation.item.created");
        send(
            &mut socket,
            serde_json::json!({ "type": "response.create" }),
        )
        .await;
        let mut saw_model_error = false;
        let mut saw_done = false;
        for _ in 0..12 {
            let event = next_event(&mut socket).await;
            match event["type"].as_str() {
                Some("error") => saw_model_error |= event["error"]["code"] == "model_not_loaded",
                Some("response.done") => {
                    saw_done = true;
                    break;
                }
                _ => {}
            }
        }
        assert!(saw_model_error, "没有收到 model_not_loaded");
        assert!(saw_done, "没有收到 response.done");
    }
}
