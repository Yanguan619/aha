//! SenseVoice ASR backend powered by sherpa-onnx.
//!
//! Model layout (downloaded from HF/ModelScope, or passed via `--weight-path`):
//! ```text
//! <dir>/model.int8.onnx   # or model.onnx
//! <dir>/tokens.txt
//! ```

use std::time::Instant;

use anyhow::{Result, anyhow};
use async_stream::stream;
use candle_core::Device;
use futures_util::Stream;
use sherpa_onnx::{
    OfflineModelConfig, OfflineRecognizer, OfflineRecognizerConfig, OfflineSenseVoiceModelConfig,
};

use crate::models::GenerateModel;
use crate::params::chat::{
    ChatCompletionChunkResponse, ChatCompletionParameters, ChatCompletionResponse,
};
use crate::utils::audio_utils::extract_audios;
use crate::utils::response_utils::{
    build_chunk_response_with_usage, build_completion_chunk_response,
    build_completion_response_with_time,
};

/// SenseVoice expects 16 kHz mono audio.
pub const SAMPLE_RATE: i32 = 16000;

pub struct SenseVoiceModel {
    recognizer: OfflineRecognizer,
    model_name: String,
}

impl SenseVoiceModel {
    /// Build a recognizer from a model directory, auto-detecting the ONNX file
    /// (`model.int8.onnx` preferred, then `model.onnx`) and `tokens.txt`.
    pub fn init_from_dir(dir: &str, num_threads: i32) -> Result<Self> {
        let dir = std::path::Path::new(dir);
        let model = ["model.int8.onnx", "model.onnx"]
            .iter()
            .map(|name| dir.join(name))
            .find(|path| path.exists())
            .ok_or_else(|| {
                anyhow!(
                    "no SenseVoice ONNX model found in {} (expected model.int8.onnx or model.onnx)",
                    dir.display()
                )
            })?;
        let tokens = dir.join("tokens.txt");
        if !tokens.exists() {
            return Err(anyhow!("missing tokens.txt in {}", dir.display()));
        }
        Self::init(
            &model.display().to_string(),
            &tokens.display().to_string(),
            num_threads,
            None,
            true,
        )
    }

    /// Build a recognizer from an already-materialized model directory.
    ///
    /// `model` is the path to `model.int8.onnx` / `model.onnx`, `tokens` the path
    /// to `tokens.txt`.
    pub fn init(
        model: &str,
        tokens: &str,
        num_threads: i32,
        language: Option<String>,
        use_itn: bool,
    ) -> Result<Self> {
        let config = OfflineRecognizerConfig {
            model_config: OfflineModelConfig {
                sense_voice: OfflineSenseVoiceModelConfig {
                    model: Some(model.to_string()),
                    // `None` lets the model auto-detect zh/en/ja/ko/yue.
                    language,
                    use_itn,
                },
                tokens: Some(tokens.to_string()),
                num_threads,
                model_type: Some("sense_voice".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };

        let recognizer = OfflineRecognizer::create(&config)
            .ok_or_else(|| anyhow!("failed to load SenseVoice model: {model}"))?;

        Ok(Self {
            recognizer,
            model_name: "sherpa-onnx-sense-voice".to_string(),
        })
    }

    /// Run one offline transcription over 16 kHz mono samples.
    fn transcribe(&self, samples: &[f32]) -> Result<String> {
        if samples.is_empty() {
            return Err(anyhow!("empty audio input"));
        }
        let stream = self.recognizer.create_stream();
        stream.accept_waveform(SAMPLE_RATE, samples);
        self.recognizer.decode(&stream);
        let result = stream
            .get_result()
            .ok_or_else(|| anyhow!("SenseVoice returned no result"))?;
        Ok(clean_sense_voice_text(&result.text))
    }

    /// Extract the first audio payload from the request and transcribe it.
    fn transcribe_messages(&mut self, mes: &ChatCompletionParameters) -> Result<String> {
        let mut audios = extract_audios(mes, &Device::Cpu, Some(SAMPLE_RATE as usize), Some(1))?;
        if audios.is_empty() {
            return Err(anyhow!("no audio input: pass an `audio_url` content part"));
        }
        let audio = audios.remove(0);
        let samples = audio.squeeze(0)?.to_vec1::<f32>()?;
        self.transcribe(&samples)
    }
}

impl GenerateModel for SenseVoiceModel {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        let started = Instant::now();
        let text = self.transcribe_messages(&mes)?;
        let secs = started.elapsed().as_secs_f64();
        Ok(build_completion_response_with_time(
            text,
            &self.model_name,
            None,
            Some(secs),
            None,
            None,
        ))
    }

    fn generate_stream(
        &mut self,
        mes: ChatCompletionParameters,
    ) -> Result<
        Box<
            dyn Stream<Item = Result<ChatCompletionChunkResponse, anyhow::Error>>
                + Send
                + Unpin
                + '_,
        >,
    > {
        let started = Instant::now();
        // SenseVoice is an offline recognizer: run it once, then emit the whole
        // transcript as a single chunk so SSE clients still work.
        let text = self.transcribe_messages(&mes)?;
        let secs = started.elapsed().as_secs_f64();
        let model_name = self.model_name.clone();
        let text_stream = stream! {
            yield Ok(build_completion_chunk_response(text, &model_name, None, None));
            yield Ok(build_chunk_response_with_usage(
                &model_name,
                Some(1),
                Some(secs),
                None,
                None,
            ));
        };
        Ok(Box::new(Box::pin(text_stream)))
    }
}

/// Strip SenseVoice rich-text tags (`<|zh|><|NEUTRAL|><|Speech|><|woitn|>...`)
/// and normalize whitespace.
///
/// Only `<|...|>` shaped tags are removed, so ordinary text such as `a < b`
/// survives untouched.
pub fn clean_sense_voice_text(raw: &str) -> String {
    if raw.is_empty() {
        return String::new();
    }
    let runes: Vec<char> = raw.chars().collect();
    let mut out = String::with_capacity(raw.len());
    let mut i = 0;
    while i < runes.len() {
        let c = runes[i];
        if c == '<' && i + 1 < runes.len() && runes[i + 1] == '|' {
            let mut j = i + 2;
            while j + 1 < runes.len() && !(runes[j] == '|' && runes[j + 1] == '>') {
                j += 1;
            }
            if j + 1 < runes.len() {
                i = j + 2;
                continue;
            }
        }
        // Drop control characters and U+FFFD produced by the model.
        if c != '\u{FFFD}' && (c >= ' ' || c == '\n' || c == '\t') {
            out.push(c);
        }
        i += 1;
    }
    let out = out.split_whitespace().collect::<Vec<_>>().join(" ");
    if out.is_empty() {
        return String::new();
    }
    // Pure punctuation is not speech.
    const PUNCT: &str = "，。！？、；：,.!?;:…—~～\t\n ";
    if out.chars().all(|c| PUNCT.contains(c)) {
        return String::new();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_strips_rich_tags() {
        assert_eq!(
            clean_sense_voice_text("<|zh|><|NEUTRAL|><|Speech|><|woitn|>今天天气不错。"),
            "今天天气不错。"
        );
    }

    #[test]
    fn test_clean_keeps_plain_angle_brackets() {
        assert_eq!(clean_sense_voice_text("a < b and c > d"), "a < b and c > d");
    }

    #[test]
    fn test_clean_punctuation_only() {
        assert_eq!(clean_sense_voice_text("，。！"), "");
    }
}
