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
#[cfg(any(feature = "sherpa", feature = "laya"))]
pub(crate) mod deskpet;
pub(crate) mod embedding;
pub(crate) mod error;
pub(crate) mod process;
pub(crate) mod reranker;

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
    let port_for_cleanup = port;
    let shutdown_handler = tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        println!("Received shutdown signal, gracefully shutting down...");
        shutdown_flag_clone.store(true, Ordering::SeqCst);
        // Give time for existing requests to complete
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        // Cleanup PID file
        let _ = cleanup_pid_file(port_for_cleanup);
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
        // OpenAI-compatible model listing endpoint: /v1/models
        .route("/v1/models", get(api::models))
        // Shutdown endpoint
        .route("/shutdown", post(api::shutdown))
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

    // 桌宠契约：/asr（裸 PCM16LE）、/tts（WAV）、/tts/stream（PCM 流）由 sherpa 后端提供，
    // /route（Laya 决策）由 laya 后端提供；没开对应 feature 时这些端点不存在。
    #[cfg(feature = "sherpa")]
    let app = app
        .route("/asr", post(deskpet::handle_asr))
        .route("/tts", post(deskpet::handle_tts))
        .route("/tts/stream", post(deskpet::handle_tts_stream));
    #[cfg(feature = "laya")]
    let app = app.route("/route", post(deskpet::handle_route));

    app
        // Unknown routes and wrong methods also answer with the OpenAI error envelope
        .fallback(error::fallback_not_found)
        .method_not_allowed_fallback(error::fallback_method_not_allowed)
        // Rewrite any framework-generated error (413/415/...) into the same envelope
        .layer(middleware::from_fn(error::normalize_error_response))
        .layer(TraceLayer::new_for_http())
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
}
