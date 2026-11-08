//! 桌宠（desktop-pet）契约端点。
//!
//! 桌宠把语音/听力/决策三条 Rust 链路都委托给 aha：同一个进程、同一个 `aha.exe`，
//! 按 `--model` 各起一个实例。这里的三个端点是桌宠 Go 侧**原有 HTTP 契约**的
//! 等价实现（换的是运行时，不是契约）：
//!
//! ```text
//! POST /asr    raw PCM16LE 16k mono  -> {text,dur_ms,took_ms}
//! POST /tts    {"text","voice"?,"speed"?} -> audio/wav (24kHz mono PCM16)
//! POST /route  {"text"} -> {target,choice,confidence,reason,text}
//! ```
//!
//! 和 aha 的 OpenAI 风格错误包不同，这三个端点失败时回 `{"error":"..."}` 的**顶层字符串**——
//! 桌宠 Go 侧解析成功体、出错只读原始 body 文本，两种都兼容，但顶层字符串最稳妥。
//!
//! 三个模型都不是并发安全的，所以处理器拿的是 **写锁**（和 `/chat` 同一条锁），
//! 一次只放一个请求进推理。

use std::time::Instant;

use axum::{
    Json,
    body::Bytes,
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
};
use serde_json::json;

use crate::server::api::MODEL;

/// ASR：单段上限 60 秒（和旧 asr-route 的 MAX_BYTES 一致）。
const ASR_MAX_BYTES: usize = 16_000 * 2 * 60;
/// ASR：小于 100ms 不值得识别。
const ASR_MIN_SAMPLES: usize = 16_000 / 10;
/// TTS：请求体上限 64KB（一句台词远到不了）。
const TTS_MAX_BODY: usize = 64 * 1024;

// --------------------------------------------------------------------------- //
// POST /asr —— 裸 PCM16LE 16k 单声道
// --------------------------------------------------------------------------- //

pub async fn handle_asr(headers: HeaderMap, body: Bytes) -> Response {
    if let Some(message) = check_asr_request(&headers, body.len()) {
        return fail(StatusCode::BAD_REQUEST, message);
    }
    // 太短：回空文本，不当错误（VAD 偶尔会切出一个极短的段）
    if body.len() < ASR_MIN_SAMPLES * 2 {
        let dur_ms = body.len() * 1000 / (16_000 * 2);
        return Json(json!({ "text": "", "dur_ms": dur_ms })).into_response();
    }

    let pcm: Vec<i16> = body
        .chunks_exact(2)
        .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
        .collect();
    let samples: Vec<f32> = pcm.iter().map(|&s| s as f32 / 32768.0).collect();
    let dur_ms = body.len() * 1000 / (16_000 * 2);

    let started = Instant::now();
    let Some(model_ref) = MODEL.get() else {
        return error_json(StatusCode::SERVICE_UNAVAILABLE, "模型没有初始化");
    };
    let guard = model_ref.write().await;
    let Some(model) = guard.instance.asr_model() else {
        return error_json(
            StatusCode::BAD_REQUEST,
            format!(
                "当前模型 {} 不是 SenseVoice，无法识别",
                guard.which_model.as_string()
            ),
        );
    };
    match model.transcribe(&samples) {
        Ok(text) => {
            let took = started.elapsed().as_millis();
            tracing::info!(dur_ms, took_ms = took, ?text, "ASR 识别完成");
            Json(json!({ "text": text, "dur_ms": dur_ms, "took_ms": took })).into_response()
        }
        Err(error) => {
            tracing::error!(%error, "ASR 识别失败");
            error_json(StatusCode::INTERNAL_SERVER_ERROR, format!("识别失败：{error:#}"))
        }
    }
}

fn check_asr_request(headers: &HeaderMap, len: usize) -> Option<String> {
    if len == 0 {
        return Some("没有音频数据".to_owned());
    }
    if len > ASR_MAX_BYTES {
        return Some(format!("音频太长（{len} 字节，上限 {ASR_MAX_BYTES}）"));
    }
    // Go 侧会带 X-Sample-Rate；缺省按 16k 处理
    let rate: i32 = headers
        .get("X-Sample-Rate")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(16_000);
    if rate != 16_000 {
        return Some(format!("只接受 16000Hz（收到 {rate}）；请在 Go 侧重采样"));
    }
    if len % 2 != 0 {
        return Some("PCM 字节数必须是偶数（16bit）".to_owned());
    }
    None
}

// --------------------------------------------------------------------------- //
// POST /tts —— {text,voice?,speed?} → audio/wav
// --------------------------------------------------------------------------- //

#[derive(serde::Deserialize)]
struct TtsRequest {
    #[serde(default)]
    text: String,
    #[serde(default)]
    voice: Option<String>,
    #[serde(default)]
    speed: Option<f32>,
}

pub async fn handle_tts(body: Bytes) -> Response {
    if body.is_empty() {
        return error_json(StatusCode::BAD_REQUEST, "请求体是空的");
    }
    if body.len() > TTS_MAX_BODY {
        return error_json(StatusCode::BAD_REQUEST, "请求体太大");
    }
    let request: TtsRequest = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(error) => return error_json(StatusCode::BAD_REQUEST, format!("JSON 不对：{error}")),
    };
    let text = request.text.trim().to_owned();
    if text.is_empty() {
        return error_json(StatusCode::BAD_REQUEST, "text 不能为空");
    }
    let voice = request.voice.unwrap_or_else(|| "zf_001".to_owned());
    let speed = request.speed.unwrap_or(1.0);

    let started = Instant::now();
    let Some(model_ref) = MODEL.get() else {
        return error_json(StatusCode::SERVICE_UNAVAILABLE, "模型没有初始化");
    };
    let guard = model_ref.write().await;
    let Some(model) = guard.instance.tts_model() else {
        return error_json(
            StatusCode::BAD_REQUEST,
            format!(
                "当前模型 {} 不是 Kokoro，无法合成",
                guard.which_model.as_string()
            ),
        );
    };
    match model.synthesize_wav(&text, &voice, speed) {
        Ok(wav) => {
            tracing::info!(bytes = wav.len(), took_ms = started.elapsed().as_millis(), "合成完成");
            ([(header::CONTENT_TYPE, "audio/wav")], wav).into_response()
        }
        Err(error) => {
            tracing::error!(%error, "合成失败");
            error_json(StatusCode::INTERNAL_SERVER_ERROR, format!("合成失败：{error:#}"))
        }
    }
}

// --------------------------------------------------------------------------- //
// POST /route —— {text} → {target,choice,confidence,reason,text}
// --------------------------------------------------------------------------- //

pub async fn handle_route(body: Bytes) -> Response {
    let text = match parse_route_body(&body) {
        Ok(text) => text,
        Err(message) => return error_json(StatusCode::BAD_REQUEST, message),
    };

    let Some(model_ref) = MODEL.get() else {
        return error_json(StatusCode::SERVICE_UNAVAILABLE, "模型没有初始化");
    };
    let guard = model_ref.write().await;
    let Some(model) = guard.instance.route_model() else {
        return error_json(
            StatusCode::BAD_REQUEST,
            format!(
                "当前模型 {} 不是 Laya，无法路由",
                guard.which_model.as_string()
            ),
        );
    };
    match model.route(&text).await {
        Ok(result) => Json(json!({
            "target": result.target,
            "choice": result.choice,
            "confidence": result.confidence,
            "reason": result.reason,
            "text": text,
        }))
        .into_response(),
        // 模型挂了要如实报错：桌宠 Go 侧据此 fail-open，别把失败伪装成 "llm"
        Err(message) => error_json(StatusCode::INTERNAL_SERVER_ERROR, message),
    }
}

fn parse_route_body(body: &[u8]) -> Result<String, String> {
    if body.is_empty() {
        return Ok(String::new());
    }
    let value: serde_json::Value =
        serde_json::from_slice(body).map_err(|_| "body 必须是 JSON {text}".to_owned())?;
    Ok(value
        .get("text")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_owned())
}

// --------------------------------------------------------------------------- //
// 小工具
// --------------------------------------------------------------------------- //

fn fail(status: StatusCode, message: impl Into<String>) -> Response {
    error_json(status, message)
}

/// 桌宠契约用的错误包：`{"error":"..."}`（顶层字符串，见文件头注释）。
fn error_json(status: StatusCode, message: impl Into<String>) -> Response {
    (status, Json(json!({ "error": message.into() }))).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    fn headers(rate: Option<&str>) -> HeaderMap {
        let mut map = HeaderMap::new();
        if let Some(rate) = rate {
            map.insert("X-Sample-Rate", HeaderValue::from_str(rate).unwrap());
        }
        map
    }

    #[test]
    fn rejects_empty_and_odd_bodies() {
        assert!(check_asr_request(&headers(None), 0).is_some());
        assert!(check_asr_request(&headers(None), 3).unwrap().contains("偶数"));
    }

    #[test]
    fn rejects_non_16k() {
        let message = check_asr_request(&headers(Some("48000")), 1000).unwrap();
        assert!(message.contains("16000"), "got {message}");
    }

    #[test]
    fn accepts_16k_and_missing_rate() {
        assert!(check_asr_request(&headers(Some("16000")), 1000).is_none());
        assert!(check_asr_request(&headers(None), 1000).is_none());
    }

    #[test]
    fn rejects_over_60_seconds() {
        assert!(check_asr_request(&headers(None), ASR_MAX_BYTES + 2).is_some());
        assert!(check_asr_request(&headers(None), ASR_MAX_BYTES).is_none());
    }

    #[test]
    fn route_body_without_text_is_empty_input() {
        assert_eq!(parse_route_body(br#"{"text":"  "}"#).unwrap(), "  ");
        assert_eq!(parse_route_body(b"").unwrap(), "");
        assert_eq!(parse_route_body(br#"{"q":"hi"}"#).unwrap(), "");
        assert!(parse_route_body(b"not json").is_err());
    }
}
