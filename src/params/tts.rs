//! `POST /v1/tts` 与 `POST /v1/tts/stream` 的请求体。
//!
//! 桌宠契约（不是 OpenAI 的 `/audio/speech`）：文本 + 音色名 + 语速。
//! 音色名对 Kokoro 是 `zf_001` 这类 id，对 GPT-SoVITS 是 `voices.json` 里的配置名。

use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub(crate) struct TtsRequest {
    /// 要合成的文本。空字符串在 handler 里会被拒绝。
    #[serde(default)]
    pub(crate) text: String,
    /// 音色名。不给时用模型自己的默认音色。
    #[serde(default)]
    pub(crate) voice: Option<String>,
    /// 语速倍率，1.0 = 原始速度。
    #[serde(default)]
    pub(crate) speed: Option<f32>,
}
