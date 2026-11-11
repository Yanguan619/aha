//! 桌宠（desktop-pet）契约端点。
//!
//! 桌宠把语音/听力/决策三条 Rust 链路都委托给 aha：一个 `aha multi` 进程按角色服务
//! 多个模型。这里的端点是桌宠 Go 侧**原有 HTTP 契约**的等价实现（换的是运行时，不是契约）：
//!
//! ```text
//! POST /v1/asr        raw PCM16LE 16k mono  -> {text,dur_ms,took_ms}
//! POST /v1/tts        {"text","voice"?,"speed"?} -> audio/wav (24kHz mono PCM16)
//! POST /v1/tts/stream {"text","voice"?,"speed"?} -> raw PCM16LE
//! POST /v1/decisions  {"text"} -> {target,choice,confidence,reason,text}
//! ```
//!
//! `/v1/decisions` 由 `laya`（System-1 类决策模型）提供。
//!
//! 和 aha 的 OpenAI 风格错误包不同，这些端点失败时回 `{"error":"..."}` 的**顶层字符串**——
//! 桌宠 Go 侧解析成功体、出错只读原始 body 文本，两种都兼容，但顶层字符串最稳妥。
//!
//! 每个模型有自己的一把写锁（见 `api::registry`）：三个角色可以并发，但同一模型
//! 一次只放一个请求进推理（sherpa/sys1 都不是并发安全的）。

#[cfg(any(feature = "sherpa-asr", feature = "sherpa-tts", feature = "gpt-sovits"))]
use std::time::Instant;

#[cfg(any(feature = "sherpa-tts", feature = "gpt-sovits"))]
use axum::body::Body;
#[cfg(any(feature = "sherpa-asr", test))]
use axum::http::HeaderMap;
use axum::{
    Json,
    body::Bytes,
    http::StatusCode,
    response::{IntoResponse, Response},
};
#[cfg(any(feature = "sherpa-tts", feature = "gpt-sovits"))]
use axum::http::header;
use serde_json::json;

use crate::params::decisions::DecisionRequest;
use crate::params::tts::TtsRequest;
use crate::server::api::model_by_key;

/// ASR：单段上限 60 秒（和旧 asr-route 的 MAX_BYTES 一致）。
/// `test` 也要算进去：`check_asr_request` 的单测不管开没开 sherpa-asr 都要跑。
#[cfg(any(feature = "sherpa-asr", test))]
const ASR_MAX_BYTES: usize = 16_000 * 2 * 60;
/// ASR：小于 100ms 不值得识别。
#[cfg(feature = "sherpa-asr")]
const ASR_MIN_SAMPLES: usize = 16_000 / 10;
/// TTS：请求体上限 64KB（一句台词远到不了）。
#[cfg(any(feature = "sherpa-tts", feature = "gpt-sovits"))]
const TTS_MAX_BODY: usize = 64 * 1024;

// --------------------------------------------------------------------------- //
// POST /v1/asr —— 裸 PCM16LE 16k 单声道
// --------------------------------------------------------------------------- //

#[cfg(feature = "sherpa-asr")]
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
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| i16::from_le_bytes(*pair))
        .collect();
    let samples: Vec<f32> = pcm.iter().map(|&s| s as f32 / 32768.0).collect();
    let dur_ms = body.len() * 1000 / (16_000 * 2);

    let started = Instant::now();
    let Some(model_ref) = model_by_key("asr") else {
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
    match tokio::task::block_in_place(|| model.transcribe(&samples)) {
        Ok(text) => {
            let took = started.elapsed().as_millis();
            tracing::info!(dur_ms, took_ms = took, ?text, "ASR 识别完成");
            Json(json!({ "text": text, "dur_ms": dur_ms, "took_ms": took })).into_response()
        }
        Err(error) => {
            tracing::error!(%error, "ASR 识别失败");
            error_json(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("识别失败：{error:#}"),
            )
        }
    }
}

#[cfg(any(feature = "sherpa-asr", test))]
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
    if !len.is_multiple_of(2) {
        return Some("PCM 字节数必须是偶数（16bit）".to_owned());
    }
    None
}

// --------------------------------------------------------------------------- //
// POST /v1/tts —— {text,voice?,speed?} → audio/wav
// --------------------------------------------------------------------------- //

#[cfg(any(feature = "sherpa-tts", feature = "gpt-sovits"))]
#[allow(clippy::result_large_err)]
fn parse_tts_request(body: &[u8]) -> Result<(String, String, f32), Response> {
    if body.is_empty() {
        return Err(error_json(StatusCode::BAD_REQUEST, "请求体是空的"));
    }
    if body.len() > TTS_MAX_BODY {
        return Err(error_json(StatusCode::BAD_REQUEST, "请求体太大"));
    }
    let request: TtsRequest = serde_json::from_slice(body)
        .map_err(|error| error_json(StatusCode::BAD_REQUEST, format!("JSON 不对：{error}")))?;
    let text = request.text.trim().to_owned();
    if text.is_empty() {
        return Err(error_json(StatusCode::BAD_REQUEST, "text 不能为空"));
    }
    let voice = request.voice.unwrap_or_else(|| "zf_001".to_owned());
    let speed = request.speed.unwrap_or(1.0);
    Ok((text, voice, speed))
}

#[cfg(any(feature = "sherpa-tts", feature = "gpt-sovits"))]
pub async fn handle_tts(body: Bytes) -> Response {
    let (text, voice, speed) = match parse_tts_request(&body) {
        Ok(parsed) => parsed,
        Err(response) => return response,
    };

    let started = Instant::now();
    let Some(model_ref) = model_by_key("tts") else {
        return error_json(StatusCode::SERVICE_UNAVAILABLE, "模型没有初始化");
    };
    let guard = model_ref.write().await;

    // GPT-SoVITS：32 kHz，音色由模型目录的 voices.json 决定
    #[cfg(feature = "gpt-sovits")]
    if let Some(sovits) = guard.instance.gpt_sovits_model() {
        let voice = sovits
            .voices()
            .contains(&voice.as_str())
            .then_some(voice.as_str());
        return match tokio::task::block_in_place(|| sovits.synthesize(&text, voice, speed)) {
            Ok(audio) => {
                let wav = match audio.squeeze(0) {
                    Ok(flat) => aha::utils::audio_utils::get_audio_wav_u8(
                        &flat,
                        sovits.sample_rate() as u32,
                    ),
                    Err(error) => Err(error.into()),
                };
                match wav {
                    Ok(wav) => {
                        tracing::info!(
                            bytes = wav.len(),
                            took_ms = started.elapsed().as_millis(),
                            "gpt-sovits 合成完成"
                        );
                        ([(header::CONTENT_TYPE, "audio/wav")], wav).into_response()
                    }
                    Err(error) => error_json(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("生成 WAV 失败：{error:#}"),
                    ),
                }
            }
            Err(error) => {
                tracing::error!(%error, "gpt-sovits 合成失败");
                error_json(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("合成失败：{error:#}"),
                )
            }
        };
    }

    #[cfg(not(feature = "sherpa-tts"))]
    {
        return error_json(
            StatusCode::BAD_REQUEST,
            format!(
                "当前模型 {} 不是 GPT-SoVITS，无法合成",
                guard.which_model.as_string()
            ),
        );
    }

    #[cfg(feature = "sherpa-tts")]
    let Some(model) = guard.instance.tts_model() else {
        return error_json(
            StatusCode::BAD_REQUEST,
            format!(
                "当前模型 {} 不是 Kokoro，无法合成",
                guard.which_model.as_string()
            ),
        );
    };
    #[cfg(feature = "sherpa-tts")]
    match tokio::task::block_in_place(|| model.synthesize_wav(&text, &voice, speed)) {
        Ok(wav) => {
            tracing::info!(
                bytes = wav.len(),
                took_ms = started.elapsed().as_millis(),
                "合成完成"
            );
            ([(header::CONTENT_TYPE, "audio/wav")], wav).into_response()
        }
        Err(error) => {
            tracing::error!(%error, "合成失败");
            error_json(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("合成失败：{error:#}"),
            )
        }
    }
}

/// `POST /v1/tts/stream` —— 边合成边吐裸 PCM16LE，桌宠 `mouth.speakStream` 用它边收边播。
///
/// 响应头：`X-Audio-Rate`（模型自报，Kokoro 24k / GPT-SoVITS 32k）、`X-Audio-Channels: 1`。
/// body：PCM16LE 小端，无 WAV 头。
///
/// 两个后端的流式机制不同但对外一致：
/// - **Kokoro（sherpa）**：声码器本身就能边合成边回调，粒度是帧；
/// - **GPT-SoVITS**：按标点切片，逐片合成，粒度是句。
///
/// 两者都在 `block_in_place` 里跑（推理是阻塞的），每块通过 channel 推给响应流。
#[cfg(any(feature = "sherpa-tts", feature = "gpt-sovits"))]
pub async fn handle_tts_stream(body: Bytes) -> Response {
    let (text, voice, speed) = match parse_tts_request(&body) {
        Ok(parsed) => parsed,
        Err(response) => return response,
    };

    let Some(model_ref) = model_by_key("tts") else {
        return error_json(StatusCode::SERVICE_UNAVAILABLE, "模型没有初始化");
    };

    // 先读一把采样率：响应头必须在 body 开始流动之前定下来。
    let sample_rate = {
        let guard = model_ref.read().await;
        match tts_sample_rate(&guard.instance) {
            Some(rate) => rate,
            None => {
                return error_json(
                    StatusCode::NOT_FOUND,
                    format!(
                        "当前模型 {} 不支持流式合成",
                        guard.which_model.as_string()
                    ),
                );
            }
        }
    };

    let (tx, rx) = tokio::sync::mpsc::channel::<Vec<u8>>(8);
    tokio::spawn(async move {
        // 写锁覆盖整段合成：合成器不是并发安全的。
        let guard = model_ref.write().await;

        #[cfg(feature = "sherpa-tts")]
        if let Some(model) = guard.instance.tts_model() {
            // Kokoro 的流式接口要求闭包 `'static`（sherpa 的 C 回调生命周期），
            // 所以这里把 tx move 进去，转换也内联。
            let tx = tx;
            let result = tokio::task::block_in_place(|| {
                model.synthesize_pcm_stream(&text, &voice, speed, move |pcm| {
                    let mut bytes = Vec::with_capacity(pcm.len() * 2);
                    for sample in pcm {
                        bytes.extend_from_slice(&sample.to_le_bytes());
                    }
                    // 接收端断了（客户端提前挂断）就停止合成。
                    tx.blocking_send(bytes).is_ok()
                })
            });
            if let Err(error) = result {
                tracing::error!(%error, "流式合成失败");
            }
            // 只有后面还跟着 gpt-sovits 的分支时才需要提前返回；
            // 只编了 sherpa-tts 时这个 return 是多余的（clippy 会报）。
            #[cfg(feature = "gpt-sovits")]
            return;
        }
        #[cfg(feature = "gpt-sovits")]
        if let Some(model) = guard.instance.gpt_sovits_model() {
            let voice = model
                .voices()
                .contains(&voice.as_str())
                .then_some(voice.as_str());
            let result = tokio::task::block_in_place(|| {
                model.synthesize_pcm_stream(&text, voice, speed, |pcm| {
                    let mut bytes = Vec::with_capacity(pcm.len() * 2);
                    for sample in pcm {
                        bytes.extend_from_slice(&sample.to_le_bytes());
                    }
                    tx.blocking_send(bytes).is_ok()
                })
            });
            if let Err(error) = result {
                tracing::error!(%error, "流式合成失败");
            }
        }
    });

    let body_stream = async_stream::stream! {
        let mut rx = rx;
        while let Some(chunk) = rx.recv().await {
            yield Ok::<_, std::io::Error>(Bytes::from(chunk));
        }
    };

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/octet-stream")
        .header("X-Audio-Rate", sample_rate.to_string())
        .header("X-Audio-Channels", "1")
        .body(Body::from_stream(body_stream))
        .expect("failed to build tts stream response")
}

// --------------------------------------------------------------------------- //
// POST /v1/decisions —— {text} → {target,choice,confidence,reason,text}
// --------------------------------------------------------------------------- //

#[cfg(feature = "laya")]
pub async fn handle_route(body: Bytes) -> Response {
    let text = match parse_route_body(&body) {
        Ok(text) => text,
        Err(message) => return error_json(StatusCode::BAD_REQUEST, message),
    };

    let Some(model_ref) = model_by_key("system1").or_else(|| model_by_key("laya")) else {
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

/// 解析 `/v1/decisions` 的请求体。
///
/// 只有 laya 后端会用到；单测也要跑（不管开没开 laya）。
#[cfg(any(feature = "laya", test))]
fn parse_route_body(body: &[u8]) -> Result<String, String> {
    if body.is_empty() {
        return Ok(String::new());
    }
    let request: DecisionRequest =
        serde_json::from_slice(body).map_err(|_| "body 必须是 JSON {text}".to_owned())?;
    Ok(request.text)
}

// --------------------------------------------------------------------------- //
// 小工具
// --------------------------------------------------------------------------- //

#[cfg(feature = "sherpa-asr")]
fn fail(status: StatusCode, message: impl Into<String>) -> Response {
    error_json(status, message)
}

/// TTS 模型的采样率（响应头用）。两个 TTS 后端都不在就是 `None`。
///
/// 原来这段是两层嵌套的 `#[cfg]` if-else（开哪几个 feature 走哪条分支），
/// 每加一个 TTS 后端就要再嵌一层。抽成一个函数，feature 只在函数体内分歧。
#[cfg(any(feature = "sherpa-tts", feature = "gpt-sovits"))]
fn tts_sample_rate(instance: &aha::models::ModelInstance<'_>) -> Option<i32> {
    #[cfg(feature = "sherpa-tts")]
    if let Some(model) = instance.tts_model() {
        return Some(model.sample_rate());
    }
    #[cfg(feature = "gpt-sovits")]
    if let Some(model) = instance.gpt_sovits_model() {
        return Some(model.sample_rate() as i32);
    }
    let _ = instance;
    None
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
        assert!(
            check_asr_request(&headers(None), 3)
                .unwrap()
                .contains("偶数")
        );
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
