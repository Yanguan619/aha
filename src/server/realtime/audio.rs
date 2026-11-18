//! 音频层：PCM16 ↔ f32、重采样、能量 VAD、发送缓冲。
//!
//! 只操作自己的字段或纯函数，不碰 socket 与模型。

use super::TurnDetection;

/// 送进 ASR 的采样率（SenseVoice / glm-asr 都吃 16k）。
pub(super) const ASR_RATE: u32 = 16_000;

/// 客户端不声明时按 24k 收（OpenAI Realtime 的默认值）。
pub(super) const DEFAULT_INPUT_RATE: u32 = 24_000;

/// 发给客户端的默认采样率。
pub(super) const DEFAULT_OUTPUT_RATE: u32 = 24_000;

/// 输入缓冲区上限：60 秒，超过就从最老的开始丢（和 `/v1/audio/transcriptions` 的上限对齐）。
pub(super) const MAX_BUFFER_SECS: u32 = 60;

/// 能量 VAD 的帧长（毫秒）。
pub(super) const VAD_FRAME_MS: u32 = 20;

/// `response.audio.delta` 的单块时长（毫秒）——太大首包慢、太小 WS 帧太碎。
pub(super) const AUDIO_CHUNK_MS: u32 = 200;

/// 客户端送进来的 PCM16 缓冲。
#[derive(Default)]
pub(super) struct AudioBuffer {
    pcm: Vec<u8>,
    rate: u32,
}

impl AudioBuffer {
    /// 缓冲区里还有没有没识别掉的音频。
    pub(super) fn is_empty(&self) -> bool {
        self.pcm.is_empty()
    }

    pub(super) fn push(&mut self, bytes: &[u8]) {
        if self.rate == 0 {
            self.rate = DEFAULT_INPUT_RATE;
        }
        self.pcm.extend_from_slice(bytes);
        let max_bytes = (u64::from(self.rate) * 2 * u64::from(MAX_BUFFER_SECS)) as usize;
        if self.pcm.len() > max_bytes {
            let drop_count = self.pcm.len() - max_bytes;
            let drop_count = drop_count + (drop_count % 2);
            self.pcm.drain(..drop_count);
        }
    }

    pub(super) fn clear(&mut self) {
        self.pcm.clear();
    }

    pub(super) fn duration_ms(&self) -> u64 {
        let rate = if self.rate == 0 {
            DEFAULT_INPUT_RATE
        } else {
            self.rate
        };
        (self.pcm.len() as u64 * 1000) / (u64::from(rate) * 2)
    }

    /// 取走缓冲（顺带清空），转成目标采样率的 f32 样本。
    pub(super) fn take_samples(&mut self, target_rate: u32) -> Vec<f32> {
        let rate = if self.rate == 0 {
            DEFAULT_INPUT_RATE
        } else {
            self.rate
        };
        let samples = pcm16_to_f32(&self.pcm);
        self.pcm.clear();
        resample_f32(&samples, rate, target_rate)
    }
}

/// PCM16LE 字节 → 归一化 f32。
pub(super) fn pcm16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| i16::from_le_bytes(*pair) as f32 / 32768.0)
        .collect()
}

/// 归一化 f32 → PCM16LE 字节。
pub(super) fn f32_to_pcm16(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        let value = (sample.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// 线性插值重采样。语音这种带宽（8k–48k）够用，不值得为它上 sinc。
pub(super) fn resample_f32(samples: &[f32], from: u32, to: u32) -> Vec<f32> {
    if samples.is_empty() || from == 0 || to == 0 || from == to {
        return samples.to_vec();
    }
    let ratio = f64::from(to) / f64::from(from);
    let out_len = ((samples.len() as f64) * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for index in 0..out_len {
        let position = index as f64 / ratio;
        let left = position.floor() as usize;
        let right = (left + 1).min(samples.len() - 1);
        let weight = (position - left as f64) as f32;
        let value = samples[left.min(samples.len() - 1)] * (1.0 - weight) + samples[right] * weight;
        out.push(value);
    }
    out
}

/// PCM16 字节直接重采样到另一个采样率（TTS 出音频用）。
pub(super) fn resample_pcm16(pcm: &[u8], from: u32, to: u32) -> Vec<u8> {
    f32_to_pcm16(&resample_f32(&pcm16_to_f32(pcm), from, to))
}

// ---------------------------------------------------------------------------
// 回合检测：能量 VAD
// ---------------------------------------------------------------------------

pub(super) enum VadEvent {
    Start,
    Stop,
}

/// 能量 VAD：按 20ms 帧算 RMS，超过阈值算说话，静音够久算说完。
///
/// 没用 FireRedVad 是因为它按"整段人声"出结果（内部要缓存到一句话结束），
/// 拿不到 `speech_started` 这种毫秒级事件；而实时对话里"开口就停"比"识别准"更要紧。
pub(super) struct EnergyVad {
    turn: TurnDetection,
    rate: u32,
    frame_samples: usize,
    frame: Vec<f32>,
    in_speech: bool,
    silence_samples: u64,
    speech_samples: u64,
}

impl EnergyVad {
    pub(super) fn new(turn: TurnDetection, rate: u32) -> Self {
        let rate = if rate == 0 { DEFAULT_INPUT_RATE } else { rate };
        let frame_samples = ((u64::from(rate) * u64::from(VAD_FRAME_MS)) / 1000).max(16) as usize;
        Self {
            turn,
            rate,
            frame_samples,
            frame: Vec::with_capacity(frame_samples),
            in_speech: false,
            silence_samples: 0,
            speech_samples: 0,
        }
    }

    pub(super) fn reset(&mut self) {
        self.frame.clear();
        self.in_speech = false;
        self.silence_samples = 0;
        self.speech_samples = 0;
    }

    pub(super) fn push(&mut self, samples: &[f32]) -> Vec<VadEvent> {
        let mut events = Vec::new();
        for sample in samples {
            self.frame.push(*sample);
            if self.frame.len() < self.frame_samples {
                continue;
            }
            let energy = self.frame.iter().map(|s| s * s).sum::<f32>() / self.frame.len() as f32;
            let voiced = energy.sqrt() >= self.turn.threshold;
            let frame_len = self.frame.len() as u64;
            if voiced {
                if !self.in_speech {
                    self.in_speech = true;
                    self.speech_samples = 0;
                    events.push(VadEvent::Start);
                }
                self.silence_samples = 0;
                self.speech_samples += frame_len;
            } else if self.in_speech {
                self.silence_samples += frame_len;
                let silence_ms = self.silence_samples * 1000 / u64::from(self.rate);
                if silence_ms >= self.turn.silence_duration_ms {
                    self.in_speech = false;
                    events.push(VadEvent::Stop);
                }
            }
            // 单句上限 30s：一直有声音也别把缓冲区撑爆。
            if self.in_speech && self.speech_samples * 1000 / u64::from(self.rate) >= 30_000 {
                self.in_speech = false;
                self.silence_samples = 0;
                events.push(VadEvent::Stop);
            }
            self.frame.clear();
        }
        events
    }
}

// ---------------------------------------------------------------------------
// 句子切分：边生成边合成，不能等整段生成完
// ---------------------------------------------------------------------------
