// OpenAI-compatible ASR (Automatic Speech Recognition) API endpoint
// Implements POST /audio/transcriptions and /v1/audio/transcriptions

use std::path::{Path, PathBuf};

use aha::models::GenerateModel;
use aha::params::chat::{
    AudioUrlType, ChatCompletionParameters, ChatMessage, ChatMessageAudioContentPart,
    ChatMessageContent, ChatMessageContentPart,
};
use aha::utils::{clean_asr_response, map_language_code};
use axum::{
    Json,
    extract::{Multipart, multipart::MultipartRejection},
    http::StatusCode,
};
use uuid::Uuid;

use crate::params::asr::{TranscriptionRequest, TranscriptionResponse};
use crate::server::api::{model_by_key, model_for};
use crate::server::error;

/// Temporary audio file written from the multipart body.
///
/// The model consumes audio through a `file://` URL, so the uploaded bytes have to
/// be materialized on disk. The file is removed when the guard is dropped.
struct TempAudioFile {
    path: PathBuf,
}

impl TempAudioFile {
    /// Create a unique path in the system temp directory, keeping the original extension
    /// so the audio loader can sniff the container format.
    fn new_path(filename: Option<&str>) -> PathBuf {
        let extension = filename
            .and_then(|name| Path::new(name).extension())
            .and_then(|ext| ext.to_str())
            .unwrap_or("wav");
        std::env::temp_dir().join(format!("aha_asr_{}.{}", Uuid::new_v4(), extension))
    }

    /// Stream a multipart field into the temp file chunk by chunk.
    ///
    /// Avoids buffering the whole upload in memory (audio files can reach 100 MiB).
    async fn write_stream(
        field: &mut axum::extract::multipart::Field<'_>,
        filename: Option<&str>,
    ) -> std::io::Result<Self> {
        use tokio::io::AsyncWriteExt;

        let path = Self::new_path(filename);
        let mut file = tokio::fs::File::create(&path).await?;
        while let Some(chunk) = field.chunk().await.map_err(std::io::Error::other)? {
            file.write_all(&chunk).await?;
        }
        file.flush().await?;
        // Close the handle before the model reads the file back.
        drop(file);
        Ok(Self { path })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempAudioFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Handle audio transcription requests
///
/// This endpoint accepts multipart/form-data with an audio file and returns
/// the transcription text in OpenAI-compatible format.
///
/// # Supported Parameters
/// - `file`: Audio file (required) - wav, mp3, m4a, etc.
/// - `model`: Model name (optional, ignored)
/// - `language`: Language code (optional) - zh, en, yue, ar, de, fr, es, pt, id, it, ko, ru, th, vi, ja, tr, hi, ms, nl, sv, da, fi, pl, cs, fil, fa, el, ro, hu, mk
/// - `prompt`: Optional prompt text (ignored in this implementation)
/// - `response_format`: Response format (only "json" supported)
/// - `temperature`: Sampling temperature (0.0 to 1.0, default 0.0)
///
/// # Returns
/// JSON response with format: `{"text": "transcribed text"}`
pub(crate) async fn transcriptions(
    multipart: Result<Multipart, MultipartRejection>,
) -> (StatusCode, Json<serde_json::Value>) {
    let mut multipart = match multipart {
        Ok(multipart) => multipart,
        Err(rejection) => {
            return error::from_multipart_rejection(rejection);
        }
    };

    let mut params = TranscriptionRequest::default();
    let mut audio_file: Option<TempAudioFile> = None;

    loop {
        let field = match multipart.next_field().await {
            Ok(Some(field)) => field,
            Ok(None) => break,
            Err(e) => {
                return error::invalid_request(
                    format!("Failed to read multipart body: {e}"),
                    None,
                    Some("invalid_multipart"),
                );
            }
        };

        let name = field.name().unwrap_or_default().to_string();
        let mut field = field;
        match name.as_str() {
            "file" => {
                let filename = field.file_name().map(|name| name.to_string());
                match TempAudioFile::write_stream(&mut field, filename.as_deref()).await {
                    Ok(file) => audio_file = Some(file),
                    Err(e) => {
                        return error::internal_error(format!("Failed to store audio file: {e}"));
                    }
                }
            }
            "model" => params.model = field.text().await.ok(),
            "language" => params.language = field.text().await.ok(),
            "prompt" => params.prompt = field.text().await.ok(),
            "response_format" => params.response_format = field.text().await.ok(),
            "temperature" => {
                params.temperature = field.text().await.ok().and_then(|v| v.parse().ok());
            }
            // Ignore unknown fields for forward compatibility
            _ => {}
        }
    }

    // Validate response_format (only JSON and text supported)
    if let Some(ref format) = params.response_format
        && format != "json"
        && format != "text"
    {
        return error::invalid_request(
            "Only 'json' and 'text' response formats are supported",
            Some("response_format"),
            Some("unsupported_format"),
        );
    }

    // Get the audio file path
    let Some(audio_file) = audio_file else {
        return error::invalid_request(
            "Audio file is required",
            Some("file"),
            Some("missing_file"),
        );
    };

    // Build file:// URL for the model
    let file_url = format!("file://{}", audio_file.path().display());

    // Map language code to full language name
    let language_name = params
        .language
        .as_ref()
        .and_then(|code| map_language_code(code));

    // Build ChatCompletionParameters for the ASR model
    let audio_part = ChatMessageContentPart::Audio(ChatMessageAudioContentPart {
        r#type: "audio".to_string(),
        audio_url: AudioUrlType { url: file_url },
    });

    let request = ChatCompletionParameters {
        messages: vec![ChatMessage::User {
            content: ChatMessageContent::ContentPart(vec![audio_part]),
            name: None,
        }],
        model: params.model.clone().unwrap_or_else(|| "asr".to_string()),
        temperature: params.temperature.or(Some(0.0)),
        max_tokens: None,
        stream: None,
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        stop: None,
        n: None,
        tools: None,
        tool_choice: None,
        response_format: None,
        metadata: language_name.map(|lang| {
            let mut map = std::collections::HashMap::new();
            map.insert("language".to_string(), lang);
            map
        }),
        ..Default::default()
    };

    // Get the model and generate transcription：优先角色 key（asr），否则按请求里的 model
    let Some(model_ref) =
        model_by_key("asr").or_else(|| model_for(params.model.as_deref()))
    else {
        return error::model_not_loaded();
    };

    let response = {
        let mut guard = model_ref.write().await;
        guard.instance.generate(request)
    };

    match response {
        Ok(chat_response) => {
            // Extract the transcription text from the response
            let raw_text = chat_response
                .choices
                .first()
                .and_then(|choice| {
                    if let ChatMessage::Assistant { content, .. } = &choice.message {
                        content.as_ref().and_then(|c| {
                            if let ChatMessageContent::Text(text) = c {
                                Some(text.clone())
                            } else {
                                None
                            }
                        })
                    } else {
                        None
                    }
                })
                .unwrap_or_else(String::new);

            // Clean the response (remove "language English<asr_text>" prefix)
            let cleaned_text = clean_asr_response(&raw_text);

            // Return OpenAI-compatible transcription response
            let transcription = TranscriptionResponse { text: cleaned_text };
            (
                StatusCode::OK,
                Json(serde_json::to_value(transcription).unwrap()),
            )
        }
        Err(e) => {
            // Determine appropriate error status based on error message
            let error_msg = e.to_string();
            if error_msg.contains("audio") || error_msg.contains("decode") {
                error::api_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    error_msg,
                    Some("file"),
                    Some("audio_decode_error"),
                )
            } else {
                error::api_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "server_error",
                    error_msg,
                    None,
                    Some("inference_error"),
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_transcription_response_serialization() {
        let response = TranscriptionResponse {
            text: "hello".to_string(),
        };
        assert_eq!(
            serde_json::to_value(response).unwrap(),
            json!({"text": "hello"})
        );
    }
}
