// ASR API data types for OpenAI-compatible transcription endpoint

use serde::Serialize;

/// Request parameters for audio transcription
///
/// The audio file itself is handled by the multipart extractor in `server::asr`;
/// every field below is parsed from the multipart form body.
#[derive(Debug, Default)]
pub(crate) struct TranscriptionRequest {
    /// ID of the model to use (ignored, always uses loaded model)
    pub(crate) model: Option<String>,

    /// Language code (e.g., "zh", "en")
    pub(crate) language: Option<String>,

    /// Optional text to guide the transcription (not implemented, ignored)
    #[allow(dead_code)]
    pub(crate) prompt: Option<String>,

    /// Response format (only "json" supported)
    pub(crate) response_format: Option<String>,

    /// Sampling temperature (0.0 to 1.0)
    pub(crate) temperature: Option<f32>,
}

/// Standard transcription response
#[derive(Debug, Serialize)]
pub(crate) struct TranscriptionResponse {
    pub(crate) text: String,
}
