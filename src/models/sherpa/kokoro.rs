//! Kokoro TTS backend powered by sherpa-onnx.
//!
//! Model layout (sherpa-onnx distribution of `hexgrad/Kokoro-82M-v1.1-zh`):
//! ```text
//! <dir>/model.onnx
//! <dir>/voices.bin
//! <dir>/tokens.txt
//! <dir>/lexicon-zh.txt      # optional
//! <dir>/dict/               # optional (jieba dictionary)
//! <dir>/espeak-ng-data/     # optional (English)
//! ```

use std::path::Path;

use anyhow::{Result, anyhow};
use async_stream::stream;
use base64::{Engine, prelude::BASE64_STANDARD};
use candle_core::{Device, Tensor};
use futures_util::Stream;
use sherpa_onnx::{
    GenerationConfig, OfflineTts, OfflineTtsConfig, OfflineTtsKokoroModelConfig,
    OfflineTtsModelConfig,
};

use crate::models::GenerateModel;
use crate::params::chat::{
    ChatCompletionChunkResponse, ChatCompletionParameters, ChatCompletionResponse,
};
use crate::utils::audio_utils::get_audio_wav_u8;
use crate::utils::response_utils::build_audio_completion_response;
use crate::utils::{extract_metadata_value, extract_user_text};

/// Default speaker id: `zf_001` in `kokoro-multi-lang-v1_1`.
pub const DEFAULT_SID: i32 = 3;

pub struct KokoroTtsModel {
    tts: OfflineTts,
    model_name: String,
    default_sid: i32,
}

fn optional(path: impl AsRef<Path>) -> Option<String> {
    let path = path.as_ref();
    path.exists().then(|| path.display().to_string())
}

impl KokoroTtsModel {
    /// Build a Kokoro engine from an already-materialized model directory.
    pub fn init(dir: &str, num_threads: i32, default_sid: i32) -> Result<Self> {
        let dir = Path::new(dir);
        let model = dir.join("model.onnx");
        let voices = dir.join("voices.bin");
        let tokens = dir.join("tokens.txt");
        for required in [&model, &voices, &tokens] {
            if !required.exists() {
                return Err(anyhow!("missing Kokoro model file: {}", required.display()));
            }
        }

        let config = OfflineTtsConfig {
            model: OfflineTtsModelConfig {
                kokoro: OfflineTtsKokoroModelConfig {
                    model: Some(model.display().to_string()),
                    voices: Some(voices.display().to_string()),
                    tokens: Some(tokens.display().to_string()),
                    lexicon: optional(dir.join("lexicon-zh.txt")),
                    dict_dir: optional(dir.join("dict")),
                    length_scale: 1.0,
                    ..Default::default()
                },
                num_threads,
                provider: None,
                ..Default::default()
            },
            ..Default::default()
        };

        let tts = OfflineTts::create(&config)
            .ok_or_else(|| anyhow!("failed to load Kokoro model: {}", dir.display()))?;

        Ok(Self {
            tts,
            model_name: "sherpa-onnx-kokoro".to_string(),
            default_sid,
        })
    }

    fn synthesize(&self, mes: &ChatCompletionParameters) -> Result<String> {
        let text = extract_user_text(mes)?;
        if text.trim().is_empty() {
            return Err(anyhow!("empty text input"));
        }
        let sid = extract_metadata_value::<i32>(&mes.metadata, "sid").unwrap_or(self.default_sid);
        let speed = extract_metadata_value::<f32>(&mes.metadata, "speed").unwrap_or(1.0);

        let config = GenerationConfig {
            sid,
            speed,
            ..Default::default()
        };
        let audio = self
            .tts
            .generate_with_config(&text, &config, None::<fn(&[f32], f32) -> bool>)
            .ok_or_else(|| anyhow!("Kokoro failed to synthesize audio"))?;

        let samples = audio.samples().to_vec();
        let sample_rate = audio.sample_rate().max(1) as u32;
        let tensor = Tensor::from_vec(samples, (1, audio.samples().len()), &Device::Cpu)?;
        let wav = get_audio_wav_u8(&tensor, sample_rate)?;
        Ok(BASE64_STANDARD.encode(wav))
    }
}

impl GenerateModel for KokoroTtsModel {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        let base64_audio = self.synthesize(&mes)?;
        Ok(build_audio_completion_response(
            &base64_audio,
            &self.model_name,
        ))
    }

    fn generate_stream(
        &mut self,
        _mes: ChatCompletionParameters,
    ) -> Result<
        Box<
            dyn Stream<Item = Result<ChatCompletionChunkResponse, anyhow::Error>>
                + Send
                + Unpin
                + '_,
        >,
    > {
        // Kokoro here is a one-shot synthesizer; streaming audio chunks is not
        // implemented. Clients should use `stream: false` or `/audio/speech`.
        let model_name = self.model_name.clone();
        let error_stream = stream! {
            yield Err(anyhow!(
                "{} does not support streaming; use stream=false or POST /audio/speech",
                model_name
            )) as Result<ChatCompletionChunkResponse, anyhow::Error>;
        };
        Ok(Box::new(Box::pin(error_stream)))
    }
}
