//! 端侧流式 ASR 会话：VAD 与识别都在引擎内完成，调用方只负责喂 PCM。
//!
//! [sokuji patch] 这是为 sokuji-android 的「进程内同传」新增的库级能力。
//!
//! 上游 aha 把"什么时候该切句"交给了调用方（`/v1/asr` 注释里写得很直白：
//! "VAD 偶尔会切出一个极短的段"），于是 Android 端只好自己做能量阈值 VAD ——
//! 环境一吵就切碎，说话一轻就丢句。既然引擎里已经有 FireRedVAD 这个流式 VAD
//! 模型，切句就该由引擎负责：
//!
//! ```text
//! 客户端: push_pcm16(帧) ─┐
//!                          │   aha::stream::AsrSession
//!                          ├─ FireRedVad::detect_frame_f32   （逐 25ms 帧）
//!                          ├─ 说话中 → 每 N ms 复解码一次 → AsrEvent::Partial
//!                          └─ 说完   → 整段送 ASR      → AsrEvent::Final
//! ```
//!
//! 复解码（re-decode）是这里"增量"的来源：GLM-ASR-Nano 这类离线声学模型没有
//! 真正的流式状态，只能对"到目前为止的音频"重新前向一遍；代价是每次 Partial 都
//! 要跑一次完整解码，所以间隔默认给到 700ms，并且只在音频够长时才做。

use crate::params::chat::{
    AudioUrlType, ChatCompletionParameters, ChatMessage, ChatMessageAudioContentPart,
    ChatMessageContent, ChatMessageContentPart,
};
use anyhow::Result;
use base64::Engine;
use candle_core::{Device, Tensor};

use crate::models::common::model_mapping::WhichModel;
use crate::models::{fire_red_vad::vad::FireRedVad, GenerateModel, ModelInstance};
use crate::utils::string_to_static_str;

/// 会话向外抛的事件。
#[derive(Debug, Clone)]
pub enum AsrEvent {
    /// 检测到开口（可用于 UI 亮"正在听"）。
    SpeechStart,
    /// 增量识别结果，可能被后续 Partial 覆盖。
    Partial(String),
    /// 一句话的最终结果。
    Final(String),
}

/// 流式识别会话。一个会话对应一路音频。
pub struct AsrSession {
    vad: FireRedVad,
    asr: ModelInstance<'static>,
    /// 还没凑够一帧（400 样本 = 25ms）的余量。
    pending: Vec<f32>,
    /// 当前这句话已经录到的音频，用来做复解码。
    speech: Vec<f32>,
    in_speech: bool,
    frame_len: usize,
    sample_rate: usize,
    prompt: String,
    max_tokens: u32,
    /// 两次 Partial 之间至少要新增多少音频（按样本数，而不是墙钟时间）。
    partial_every_samples: usize,
    /// 下一帧 Partial 最早在缓冲区多长时触发。
    next_partial_at: usize,
    /// 音频短于这个长度就不做 Partial（解码纯浪费）。
    partial_min_samples: usize,
    /// 缓冲涨到这么长才认为"开口了"，用于发 SpeechStart。
    speech_start_samples: usize,
    /// 单句上限，超过就强制收尾，避免无限增长。
    max_speech_samples: usize,
}

impl AsrSession {
    /// `vad_dir` 需包含 `cmvn.json` + `*.safetensors`，且**目录名要带 `stream`**
    /// （FireRedVad 用目录名区分流式/离线两套配置）。
    pub fn new(vad_dir: &str, asr_model: WhichModel, asr_dir: &str) -> Result<Self> {
        let device = Device::Cpu;
        let vad = FireRedVad::init(vad_dir, Some(&device), None)?;
        let asr = crate::models::load_model(
            asr_model,
            string_to_static_str(asr_dir.to_string()),
            Some(&Device::Cpu),
            None,
            4,
        )?;
        Ok(Self {
            vad,
            asr,
            pending: Vec::new(),
            speech: Vec::new(),
            in_speech: false,
            frame_len: 400,
            sample_rate: 16_000,
            prompt: "Please transcribe this audio into text".to_string(),
            max_tokens: 256,
            partial_every_samples: 16_000 * 700 / 1000,
            next_partial_at: 16_000,
            partial_min_samples: 16_000,  // 1s
            speech_start_samples: 4_800,  // 300ms
            max_speech_samples: 16_000 * 20, // 20s
        })
    }

    /// 调整增量识别的节奏；传 0 表示关掉 Partial（只要 Final）。
    pub fn set_partial_interval_ms(&mut self, ms: u64) {
        self.partial_every_samples = self.sample_rate * ms as usize / 1000;
    }

    /// 喂一段 PCM16LE 单声道音频，返回这一批产生的事件。
    ///
    /// `sample_rate` 可以是 24000（App 采集的原生采样率），内部会重采样到 16k。
    pub fn push_pcm16(&mut self, pcm: &[u8], sample_rate: u32) -> Result<Vec<AsrEvent>> {
        let mut events = Vec::new();
        if pcm.len() < 2 {
            return Ok(events);
        }
        let samples: Vec<f32> = pcm
            .chunks_exact(2)
            .map(|pair| i16::from_le_bytes([pair[0], pair[1]]) as f32 / 32768.0)
            .collect();

        let samples = if sample_rate as usize != self.sample_rate {
            let tensor = Tensor::from_vec(samples, (1, pcm.len() / 2), &Device::Cpu)?;
            let tensor = crate::utils::audio_utils::resample_simple(
                &tensor,
                sample_rate as i64,
                self.sample_rate as i64,
            )?;
            tensor.flatten_all()?.to_vec1::<f32>()?
        } else {
            samples
        };

        self.pending.extend_from_slice(&samples);

        while self.pending.len() >= self.frame_len {
            let frame: Vec<f32> = self.pending.drain(..self.frame_len).collect();
            // 每一帧都进缓冲。VAD 只在"一句话说完"时把整段音频还回来，说话过程中的
            // 增量文本得靠这份缓冲反复解码 —— 之前只在 in_speech 时才缓存，而 in_speech
            // 又依赖缓冲非空，结果 Partial 永远触发不了。
            self.speech.extend_from_slice(&frame);
            let vad_out = self
                .vad
                .detect_frame_f32(frame, 1, Some(self.sample_rate))?;
            match vad_out {
                Some(result) => {
                    // 一句话说完了：VAD 把缓存的音频整段还给我们。
                    let audio = match result.orig_audio {
                        Some(tensor) => tensor.flatten_all()?.to_vec1::<f32>()?,
                        None => std::mem::take(&mut self.speech),
                    };
                    self.in_speech = false;
                    self.speech.clear();
                    self.next_partial_at = self.partial_min_samples;
                    if audio.len() >= self.sample_rate / 5 {
                        let text = self.transcribe(&audio)?;
                        if !text.is_empty() {
                            events.push(AsrEvent::Final(text));
                        }
                    }
                }
                None => {
                    if !self.in_speech && self.speech.len() >= self.speech_start_samples {
                        self.in_speech = true;
                        events.push(AsrEvent::SpeechStart);
                    }
                    if self.in_speech {
                        if self.speech.len() >= self.max_speech_samples {
                            let audio = std::mem::take(&mut self.speech);
                            self.in_speech = false;
                            self.next_partial_at = self.partial_min_samples;
                            let text = self.transcribe(&audio)?;
                            if !text.is_empty() {
                                events.push(AsrEvent::Final(text));
                            }
                            continue;
                        }
                        // 按"新增了多少音频"而不是墙钟时间来节流：一次解码可能比它覆盖的
                        // 音频还慢，用时间做闸门会让 Partial 连成一串。
                        if self.partial_every_samples > 0
                            && self.speech.len() >= self.partial_min_samples
                            && self.speech.len() >= self.next_partial_at
                        {
                            let text = self.transcribe(&self.speech.clone())?;
                            self.next_partial_at = self.speech.len() + self.partial_every_samples;
                            if !text.is_empty() {
                                events.push(AsrEvent::Partial(text));
                            }
                        }
                    }
                }
            }
        }
        Ok(events)
    }

    /// 强制收尾：把手上没有说完的音频也识别掉（用户点停止时调用）。
    pub fn flush(&mut self) -> Result<Option<String>> {
        let audio = std::mem::take(&mut self.speech);
        self.pending.clear();
        self.in_speech = false;
        self.next_partial_at = self.partial_min_samples;
        self.vad.reset();
        if audio.len() < self.sample_rate / 5 {
            return Ok(None);
        }
        let text = self.transcribe(&audio)?;
        Ok(if text.is_empty() { None } else { Some(text) })
    }

    pub fn reset(&mut self) {
        self.pending.clear();
        self.speech.clear();
        self.in_speech = false;
        self.next_partial_at = self.partial_min_samples;
        self.vad.reset();
    }

    /// 把一段 f32 音频封成 WAV → base64 data URL → 走一次 ASR。
    fn transcribe(&mut self, audio: &[f32]) -> Result<String> {
        let wav = wav_from_f32(audio, self.sample_rate as u32);
        let url = format!(
            "data:audio/wav;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(wav)
        );
        let request = ChatCompletionParameters {
            messages: vec![ChatMessage::User {
                content: ChatMessageContent::ContentPart(vec![
                    ChatMessageContentPart::Audio(ChatMessageAudioContentPart {
                        r#type: "audio".to_string(),
                        audio_url: AudioUrlType { url },
                    }),
                    ChatMessageContentPart::Text(
                        crate::params::chat::ChatMessageTextContentPart {
                            r#type: "text".to_string(),
                            text: self.prompt.clone(),
                        },
                    ),
                ]),
                name: None,
            }],
            model: "asr".to_string(),
            max_tokens: Some(self.max_tokens),
            temperature: Some(0.0),
            ..Default::default()
        };
        let response = self.asr.generate(request)?;
        let raw = response
            .choices
            .first()
            .and_then(|choice| match &choice.message {
                ChatMessage::Assistant {
                    content: Some(ChatMessageContent::Text(text)),
                    ..
                } => Some(text.clone()),
                _ => None,
            })
            .unwrap_or_default();
        Ok(clean_asr_text(&raw))
    }
}

/// GLM-ASR 有时会输出 `language Chinese<asr_text>……` 这种前缀，去掉它。
fn clean_asr_text(raw: &str) -> String {
    let text = match raw.split_once("<asr_text>") {
        Some((_, rest)) => rest,
        None => raw,
    };
    text.trim().to_string()
}

/// f32 [-1,1] → 16bit 单声道 WAV bytes。
fn wav_from_f32(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let data_len = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for &sample in samples {
        let value = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}
