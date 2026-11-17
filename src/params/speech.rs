//! `POST /audio/speech`（OpenAI 形状）与桌宠 `/v1/tts` 的请求体。
//!
//! 桌宠那两条别名端点也用这个结构：`input` 对 `text`、`response_format` 决定
//! 返回 WAV 还是 PCM 流。桌宠 `/v1/tts` 的 `{text,voice,speed}` 形状由
//! `params::tts::TtsRequest` 单独保留（那条契约不改）。

use serde::Deserialize;

#[derive(Debug, Default, Deserialize)]
pub(crate) struct SpeechRequest {
    /// 模型名。和 ASR 端点一样被忽略：始终用已加载的那个 TTS。
    #[serde(default)]
    #[allow(dead_code)]
    pub(crate) model: Option<String>,
    /// 要合成的文本。
    #[serde(default)]
    pub(crate) input: String,
    /// 音色名。不给时用模型自己的默认音色。
    #[serde(default)]
    pub(crate) voice: Option<String>,
    /// `wav`（默认）或 `pcm`。`pcm` 走 chunked 流，边合成边推。
    #[serde(default)]
    pub(crate) response_format: Option<String>,
    /// 语速倍率，1.0 = 原始速度（OpenAI 的范围是 0.25–4.0）。
    #[serde(default)]
    pub(crate) speed: Option<f32>,
}
