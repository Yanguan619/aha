//! TTS 的通用实现：OpenAI 形状的 `POST /audio/speech` 在这里。
//!
//! 桌宠那两条别名端点（`/v1/tts` 回 WAV、`/v1/tts/stream` 回裸 PCM 流）只是
//! 解析各自的请求体，然后调这里的 `synthesize_wav` / `stream_pcm`——实现只有一份。
//!
//! 两个后端（Kokoro / GPT-SoVITS）的差异只在这两个函数内部按模型分支。

use std::time::Instant;

use axum::{
    body::{Body, Bytes},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};

use crate::params::speech::SpeechRequest;
use crate::server::api::model_by_key;
use crate::server::error;

/// TTS 请求体上限 64KB（一句台词远到不了）。
pub(crate) const TTS_MAX_BODY: usize = 64 * 1024;
/// 默认音色：Kokoro 的 `zf_001`。GPT-SoVITS 里没有这个名字时由模型挑自己的默认音色。
const DEFAULT_VOICE: &str = "zf_001";

/// 内部失败：状态码 + 文案，由 [`speech`] 包成 OpenAI envelope。
pub(crate) type TtsFail = (StatusCode, String);

/// 整段合成 → WAV 字节（24k Kokoro / 32k GPT-SoVITS，各自在 WAV 头里）。
pub(crate) async fn synthesize_wav(
    text: &str,
    voice: &str,
    speed: f32,
) -> Result<Vec<u8>, TtsFail> {
    let started = Instant::now();
    let Some(model_ref) = model_by_key("tts") else {
        return Err((StatusCode::SERVICE_UNAVAILABLE, "模型没有初始化".to_owned()));
    };
    let guard = model_ref.write().await;

    // GPT-SoVITS：音色由模型目录的 voices.json 决定
    #[cfg(feature = "gpt-sovits")]
    if let Some(sovits) = guard.instance.gpt_sovits_model() {
        let voice = sovits.voices().contains(&voice).then_some(voice);
        let audio = tokio::task::block_in_place(|| sovits.synthesize(text, voice, speed))
            .map_err(|error| synth_failed(&error))?;
        let flat = audio.squeeze(0).map_err(|error| wav_failed(&error))?;
        let wav = aha::utils::audio_utils::get_audio_wav_u8(&flat, sovits.sample_rate() as u32)
            .map_err(|error| wav_failed(&error))?;
        tracing::info!(
            bytes = wav.len(),
            took_ms = started.elapsed().as_millis(),
            "gpt-sovits 合成完成"
        );
        return Ok(wav);
    }

    #[cfg(feature = "sherpa-tts")]
    if let Some(model) = guard.instance.tts_model() {
        let wav = tokio::task::block_in_place(|| model.synthesize_wav(text, voice, speed))
            .map_err(|error| synth_failed(&error))?;
        tracing::info!(
            bytes = wav.len(),
            took_ms = started.elapsed().as_millis(),
            "合成完成"
        );
        return Ok(wav);
    }

    Err((
        StatusCode::BAD_REQUEST,
        format!("当前模型 {} 无法合成", guard.which_model.as_string()),
    ))
}

/// 边合成边吐裸 PCM16LE（无 WAV 头）。响应头 `X-Audio-Rate` / `X-Audio-Channels`。
///
/// 两个后端的流式机制不同但对外一致：
/// - **Kokoro（sherpa）**：声码器本身就能边合成边回调，粒度是帧；
/// - **GPT-SoVITS**：按标点切片，逐片合成，粒度是句。
///
/// 两者都在 `block_in_place` 里跑（推理是阻塞的），每块通过 channel 推给响应流。
pub(crate) async fn stream_pcm(
    text: String,
    voice: String,
    speed: f32,
) -> Result<Response, TtsFail> {
    let Some(model_ref) = model_by_key("tts") else {
        return Err((StatusCode::SERVICE_UNAVAILABLE, "模型没有初始化".to_owned()));
    };

    // 先读一把采样率：响应头必须在 body 开始流动之前定下来。
    let sample_rate = {
        let guard = model_ref.read().await;
        sample_rate_of(&guard.instance).ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                format!("当前模型 {} 不支持流式合成", guard.which_model.as_string()),
            )
        })?
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
                    // 接收端断了（客户端提前挂断）就停止合成。
                    tx.blocking_send(pcm_to_bytes(pcm)).is_ok()
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
                    tx.blocking_send(pcm_to_bytes(pcm)).is_ok()
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

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/octet-stream")
        .header("X-Audio-Rate", sample_rate.to_string())
        .header("X-Audio-Channels", "1")
        .body(Body::from_stream(body_stream))
        .expect("failed to build tts stream response"))
}

/// `POST /audio/speech`——OpenAI 形状的 TTS。
///
/// 请求 `{model,input,voice?,response_format?,speed?}`；`model` 和 ASR 端点一样被忽略。
/// `response_format` 只支持 `wav`（默认，整段）和 `pcm`（chunked 裸流）；
/// 其余格式（mp3/opus/flac/aac）如实返回 400，不静默降级成别的编码。
pub(crate) async fn speech(body: Bytes) -> Response {
    let (text, voice, speed, format) = match parse_speech(&body) {
        Ok(parsed) => parsed,
        Err((status, message)) => {
            return error::api_error(
                status,
                "invalid_request_error",
                message,
                None,
                Some("invalid_request"),
            )
            .into_response();
        }
    };

    let result = match format.as_str() {
        "pcm" => stream_pcm(text, voice, speed).await,
        // parse_speech 已经把白名单收窄到 wav / pcm
        _ => synthesize_wav(&text, &voice, speed)
            .await
            .map(|wav| ([(header::CONTENT_TYPE, "audio/wav")], wav).into_response()),
    };

    match result {
        Ok(response) => response,
        Err((status, message)) => {
            let error_type = if status == StatusCode::INTERNAL_SERVER_ERROR {
                "server_error"
            } else {
                "invalid_request_error"
            };
            error::api_error(status, error_type, message, None, None).into_response()
        }
    }
}

/// 解析 `/audio/speech` 的请求体 → `(text, voice, speed, response_format)`。
///
/// 纯函数（不碰模型），单测直接打它。
fn parse_speech(body: &[u8]) -> Result<(String, String, f32, String), (StatusCode, String)> {
    if body.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "请求体是空的".to_owned()));
    }
    if body.len() > TTS_MAX_BODY {
        return Err((StatusCode::BAD_REQUEST, "请求体太大".to_owned()));
    }
    let request: SpeechRequest = serde_json::from_slice(body)
        .map_err(|error| (StatusCode::BAD_REQUEST, format!("JSON 不对：{error}")))?;
    let text = request.input.trim().to_owned();
    if text.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "input 不能为空".to_owned()));
    }
    let speed = request.speed.unwrap_or(1.0);
    if !(0.25..=4.0).contains(&speed) {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("speed 必须在 0.25–4.0（收到 {speed}）"),
        ));
    }
    let format = request
        .response_format
        .unwrap_or_else(|| "wav".to_owned())
        .to_ascii_lowercase();
    if format != "wav" && format != "pcm" {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("只支持 response_format = wav / pcm（收到 {format}）"),
        ));
    }
    let voice = request
        .voice
        .filter(|voice| !voice.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_VOICE.to_owned());
    Ok((text, voice, speed, format))
}

/// 整段合成 → 裸 PCM16LE + 采样率（不要 WAV 头）。
///
/// `/realtime` 用这个：它把音频当 base64 塞事件里，自己按 200ms 切块，
/// 所以只要样本和采样率，WAV 头反而是噪音。实现就是走一遍 [`synthesize_wav`]
/// 再把 WAV 解析回来——两条路径的模型分支逻辑只保留一份。
/// 当前 TTS 模型的采样率（Hz）。`/realtime` 要在开始推块之前就知道它。
pub(crate) async fn sample_rate() -> Option<u32> {
    let model_ref = model_by_key("tts")?;
    let guard = model_ref.read().await;
    sample_rate_of(&guard.instance).map(|rate| rate as u32)
}

/// 边合成边把 PCM16LE 块推给 `sink`（首包延迟 ≈ 帧级，不是整句）。
///
/// `sink` 满了会阻塞合成线程（背压）；接收端关掉（客户端走了）就中止合成。
/// 返回模型的采样率——调用方按需重采样到会话的输出率。
pub(crate) async fn synthesize_pcm16_stream(
    text: String,
    voice: String,
    speed: f32,
    sink: tokio::sync::mpsc::Sender<Vec<u8>>,
) -> Result<u32, TtsFail> {
    let Some(model_ref) = model_by_key("tts") else {
        return Err((StatusCode::SERVICE_UNAVAILABLE, "模型没有初始化".to_owned()));
    };
    let sample_rate = {
        let guard = model_ref.read().await;
        sample_rate_of(&guard.instance).ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                format!("当前模型 {} 不支持流式合成", guard.which_model.as_string()),
            )
        })? as u32
    };
    let guard = model_ref.write().await;

    #[cfg(feature = "sherpa-tts")]
    if let Some(model) = guard.instance.tts_model() {
        let sink = sink;
        let result = tokio::task::block_in_place(|| {
            model.synthesize_pcm_stream(&text, &voice, speed, move |pcm| {
                sink.blocking_send(pcm_to_bytes(pcm)).is_ok()
            })
        });
        return result
            .map(|_| sample_rate)
            .map_err(|error| synth_failed(&error));
    }
    #[cfg(feature = "gpt-sovits")]
    if let Some(model) = guard.instance.gpt_sovits_model() {
        let voice = model
            .voices()
            .contains(&voice.as_str())
            .then_some(voice.as_str());
        let sink = sink;
        let result = tokio::task::block_in_place(|| {
            model.synthesize_pcm_stream(&text, voice, speed, move |pcm| {
                sink.blocking_send(pcm_to_bytes(pcm)).is_ok()
            })
        });
        return result
            .map(|_| sample_rate)
            .map_err(|error| synth_failed(&error));
    }

    Err((
        StatusCode::BAD_REQUEST,
        format!("当前模型 {} 无法合成", guard.which_model.as_string()),
    ))
}

fn pcm_to_bytes(pcm: &[i16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(pcm.len() * 2);
    for sample in pcm {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}

fn synth_failed(error: &impl std::fmt::Display) -> TtsFail {
    tracing::error!(%error, "合成失败");
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        format!("合成失败：{error:#}"),
    )
}

#[cfg(feature = "gpt-sovits")]
fn wav_failed(error: &impl std::fmt::Display) -> TtsFail {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        format!("生成 WAV 失败：{error:#}"),
    )
}

/// TTS 模型的采样率（流式响应头用）。两个 TTS 后端都不在就是 `None`。
///
/// 原来这段是两层嵌套的 `#[cfg]` if-else（开哪几个 feature 走哪条分支），
/// 每加一个 TTS 后端就要再嵌一层。抽成一个函数，feature 只在函数体内分歧。
fn sample_rate_of(instance: &aha::models::ModelInstance<'_>) -> Option<i32> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speech_defaults_to_wav_and_default_voice() {
        let (text, voice, speed, format) =
            parse_speech(r#"{"input":" 你好 "}"#.as_bytes()).unwrap();
        assert_eq!(text, "你好");
        assert_eq!(voice, DEFAULT_VOICE);
        assert_eq!(speed, 1.0);
        assert_eq!(format, "wav");
    }

    #[test]
    fn speech_accepts_pcm_and_rejects_other_formats() {
        let (_, _, _, format) =
            parse_speech(r#"{"input":"嗨","response_format":"PCM"}"#.as_bytes()).unwrap();
        assert_eq!(format, "pcm");
        let message = parse_speech(r#"{"input":"嗨","response_format":"mp3"}"#.as_bytes())
            .unwrap_err()
            .1;
        assert!(message.contains("wav"), "got {message}");
    }

    #[test]
    fn speech_rejects_empty_input_and_bad_speed() {
        assert!(parse_speech(r#"{"input":"   "}"#.as_bytes()).is_err());
        assert!(parse_speech(b"").is_err());
        assert!(parse_speech(r#"{"input":"嗨","speed":9.0}"#.as_bytes()).is_err());
        assert!(parse_speech(b"not json").is_err());
    }
}
