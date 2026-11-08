//! Kokoro TTS backend powered by sherpa-onnx.
//!
//! Model layout (sherpa-onnx distribution of `hexgrad/Kokoro-82M-v1.1-zh`):
//! ```text
//! <dir>/model.onnx
//! <dir>/voices.bin
//! <dir>/tokens.txt
//! <dir>/lexicon-us-en.txt   # optional
//! <dir>/lexicon-zh.txt      # optional
//! <dir>/dict/               # optional (jieba dictionary)
//! <dir>/espeak-ng-data/     # optional (English OOV fallback)
//! ```
//!
//! 除了 OpenAI 风格的 `/audio/speech`，本模块还给桌宠那条更小的契约
//! （`POST /tts` → 裸 WAV）提供 [`KokoroTtsModel::synthesize_wav`]：只有一份模型、
//! 一套音色名（`zf_001` …），换运行时不该换音色。

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

/// 一句台词的上限：调用方已经按句切过，这里只是兜底（超了截断，不是报错）。
pub const MAX_TEXT: usize = 300;

/// 音色名按 sid 排列（v1_1 的 103 个音色）。索引 = sid。
/// 来源：<https://k2-fsa.github.io/sherpa/onnx/tts/all/Chinese-English/kokoro-multi-lang-v1_1.html>
pub const VOICES: &[&str] = &[
    "af_maple", "af_sol", "bf_vale", // 0-2
    "zf_001", "zf_002", "zf_003", "zf_004", "zf_005", "zf_006", "zf_007", "zf_008", // 3-10
    "zf_017", "zf_018", "zf_019", "zf_021", "zf_022", "zf_023", "zf_024", "zf_026", // 11-18
    "zf_027", "zf_028", "zf_032", "zf_036", "zf_038", "zf_039", "zf_040", "zf_042", // 19-26
    "zf_043", "zf_044", "zf_046", "zf_047", "zf_048", "zf_049", "zf_051", "zf_059", // 27-34
    "zf_060", "zf_067", "zf_070", "zf_071", "zf_072", "zf_073", "zf_074", "zf_075", // 35-42
    "zf_076", "zf_077", "zf_078", "zf_079", "zf_083", "zf_084", "zf_085", "zf_086", // 43-50
    "zf_087", "zf_088", "zf_090", "zf_092", "zf_093", "zf_094", "zf_099", // 51-57
    "zm_009", "zm_010", "zm_011", "zm_012", "zm_013", "zm_014", "zm_015", "zm_016", // 58-65
    "zm_020", "zm_025", "zm_029", "zm_030", "zm_031", "zm_033", "zm_034", "zm_035", // 66-73
    "zm_037", "zm_041", "zm_045", "zm_050", "zm_052", "zm_053", "zm_054", "zm_055", // 74-81
    "zm_056", "zm_057", "zm_058", "zm_061", "zm_062", "zm_063", "zm_064", "zm_065", // 82-89
    "zm_066", "zm_068", "zm_069", "zm_080", "zm_081", "zm_082", "zm_089", "zm_091", // 90-97
    "zm_095", "zm_096", "zm_097", "zm_098", "zm_100", // 98-102
];

/// 音色名 → sid。支持三种写法：`zf_001`、纯数字、空（调用方处理默认值）。
pub fn resolve_voice(voice: &str) -> Option<i32> {
    let voice = voice.trim();
    if voice.is_empty() {
        return None;
    }
    if let Ok(sid) = voice.parse::<i32>() {
        return (0..VOICES.len() as i32).contains(&sid).then_some(sid);
    }
    VOICES
        .iter()
        .position(|name| name.eq_ignore_ascii_case(voice))
        .map(|index| index as i32)
}

/// 按字符（不是字节）截断——中文一个字三字节，按字节切会切出半个字。
fn truncate(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_owned();
    }
    text.chars().take(max).collect()
}

pub struct KokoroTtsModel {
    tts: OfflineTts,
    model_name: String,
    default_sid: i32,
    model_dir: String,
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

        // 中文多音字/数字靠 lexicon + jieba 词典；英文 OOV 靠 espeak（默认不下载）。
        // 哪个存在就传哪个：传不存在的路径会让创建阶段直接失败。
        let lexicons: Vec<String> = ["lexicon-us-en.txt", "lexicon-zh.txt"]
            .iter()
            .filter_map(|name| optional(dir.join(name)))
            .collect();
        let lexicon = (!lexicons.is_empty()).then(|| lexicons.join(","));
        let espeak = dir.join("espeak-ng-data");
        let data_dir = espeak.is_dir().then(|| espeak.display().to_string());

        let config = OfflineTtsConfig {
            model: OfflineTtsModelConfig {
                kokoro: OfflineTtsKokoroModelConfig {
                    model: Some(model.display().to_string()),
                    voices: Some(voices.display().to_string()),
                    tokens: Some(tokens.display().to_string()),
                    lexicon,
                    dict_dir: optional(dir.join("dict")),
                    data_dir,
                    length_scale: 1.0,
                    ..Default::default()
                },
                num_threads,
                provider: Some("cpu".to_owned()),
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
            model_dir: dir.display().to_string(),
        })
    }

    pub fn default_sid(&self) -> i32 {
        self.default_sid
    }

    pub fn model_dir(&self) -> &str {
        &self.model_dir
    }

    /// 模型自报的采样率（Kokoro 是 24k；WAV 头按它写，不能写死）。
    pub fn sample_rate(&self) -> i32 {
        self.tts.sample_rate().max(1)
    }

    pub fn num_speakers(&self) -> i32 {
        self.tts.num_speakers()
    }

    /// 一句话 → 完整 WAV（24kHz mono PCM16）。桌宠的 `POST /tts` 直接回这个字节流。
    ///
    /// `voice` 可以是 `zf_001` 这种名字、也可以是数字 sid；空 = 默认音色。
    /// **合成器不是并发安全的**，调用方必须串行（`/tts` 走 aha 的写锁）。
    pub fn synthesize_wav(&self, text: &str, voice: &str, speed: f32) -> Result<Vec<u8>> {
        let text = truncate(text, MAX_TEXT);
        if text.trim().is_empty() {
            return Err(anyhow!("text 不能为空"));
        }
        let sid = if voice.trim().is_empty() {
            self.default_sid
        } else {
            resolve_voice(voice).unwrap_or_else(|| {
                tracing::warn!(voice, sid = self.default_sid, "认不出这个音色，用默认");
                self.default_sid
            })
        };
        let config = GenerationConfig {
            sid,
            speed: if speed > 0.0 { speed } else { 1.0 },
            ..Default::default()
        };
        let audio = self
            .tts
            .generate_with_config(&text, &config, None::<fn(&[f32], f32) -> bool>)
            .ok_or_else(|| anyhow!("Kokoro failed to synthesize audio"))?;

        let samples = audio.samples().to_vec();
        if samples.is_empty() {
            return Err(anyhow!("合成结果为空"));
        }
        let sample_rate = audio.sample_rate().max(1) as u32;
        let tensor = Tensor::from_vec(samples, (1, audio.samples().len()), &Device::Cpu)?;
        get_audio_wav_u8(&tensor, sample_rate)
    }

    /// 边合成边吐 **PCM16LE**（桌宠 `POST /tts/stream` 用）。
    ///
    /// 回调拿到的是**本次新增**的样本（sherpa 的 C 回调语义），返回 `false` 可以中止合成。
    /// 和 [`Self::synthesize_wav`] 一样，合成器不是并发安全的，调用方必须串行。
    pub fn synthesize_pcm_stream<F>(
        &self,
        text: &str,
        voice: &str,
        speed: f32,
        mut emit: F,
    ) -> Result<()>
    where
        F: FnMut(&[i16]) -> bool + 'static,
    {
        let text = truncate(text, MAX_TEXT);
        if text.trim().is_empty() {
            return Err(anyhow!("text 不能为空"));
        }
        let sid = if voice.trim().is_empty() {
            self.default_sid
        } else {
            resolve_voice(voice).unwrap_or_else(|| {
                tracing::warn!(voice, sid = self.default_sid, "认不出这个音色，用默认");
                self.default_sid
            })
        };
        let config = GenerationConfig {
            sid,
            speed: if speed > 0.0 { speed } else { 1.0 },
            ..Default::default()
        };

        let mut buffer: Vec<i16> = Vec::new();
        let callback = move |samples: &[f32], _progress: f32| -> bool {
            buffer.clear();
            buffer.extend(
                samples
                    .iter()
                    .map(|sample| (sample.clamp(-1.0, 1.0) * 32767.0).round() as i16),
            );
            emit(&buffer)
        };

        let audio = self
            .tts
            .generate_with_config(&text, &config, Some(callback))
            .ok_or_else(|| anyhow!("Kokoro failed to synthesize audio"))?;
        if audio.samples().is_empty() {
            return Err(anyhow!("合成结果为空"));
        }
        Ok(())
    }

    /// 空跑一句，把第一帧的冷启动挪到"端口开着之前"。
    pub fn warmup(&self) {
        if let Err(error) = self.synthesize_wav("嗯", "", 1.0) {
            tracing::warn!(%error, "Kokoro 预热失败");
        }
    }

    fn synthesize(&self, mes: &ChatCompletionParameters) -> Result<String> {
        let text = extract_user_text(mes)?;
        let sid = extract_metadata_value::<i32>(&mes.metadata, "sid").unwrap_or(self.default_sid);
        let speed = extract_metadata_value::<f32>(&mes.metadata, "speed").unwrap_or(1.0);
        // 走同一个入口：WAV 字节 → base64（OpenAI 风格响应用）
        let wav = self.synthesize_wav(&text, &sid.to_string(), speed)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voice_table_matches_the_model() {
        // 官方表：103 个音色，0-2 英文、3-57 中文女声、58-102 中文男声
        assert_eq!(VOICES.len(), 103);
        assert_eq!(resolve_voice("zf_001"), Some(3));
        assert_eq!(resolve_voice("ZF_001"), Some(3));
        assert_eq!(resolve_voice("zm_100"), Some(102));
        assert_eq!(resolve_voice("3"), Some(3));
        assert_eq!(resolve_voice("zf_999"), None);
        assert_eq!(resolve_voice("103"), None);
        assert_eq!(resolve_voice(""), None);
    }

    #[test]
    fn truncates_by_chars_not_bytes() {
        let text = "好".repeat(400);
        let cut = truncate(&text, MAX_TEXT);
        assert_eq!(cut.chars().count(), MAX_TEXT);
        assert!(!cut.contains('\u{FFFD}'));
    }
}
