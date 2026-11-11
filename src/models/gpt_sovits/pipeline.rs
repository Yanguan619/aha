//! GPT-SoVITS 端到端推理（文本 + 参考音频 → 32 kHz 波形）。
//!
//! 一条 `GptSovitsModel` 里装着 5 个组件：
//!
//! ```text
//! ZhFrontend（前端） → RobertaEncoder（文本特征） → Text2Semantic（AR）
//!                    ↗ Hubert（参考音频内容特征）  → Sovits（量化 + 合成 → wav）
//! ```
//!
//! 音色通过模型目录里的 `voices.json` 描述：
//!
//! ```json
//! { "default": { "ref_audio": "ref.wav", "speed": 1.0 } }
//! ```
//!
//! 没写 `voices.json` 时退化到目录下的 `ref.wav`。

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use serde::Deserialize;

use crate::utils::audio_utils::{load_audio_use_hound, load_audio_use_symphonia, resample_simple};

use super::hubert::Hubert;
use super::roberta::RobertaEncoder;
use super::sovits::{Sovits, SynthOptions};
use super::style::linear_spectrogram;
use super::t2s::{GenOptions, Text2Semantic};
use super::text_frontend::ZhFrontend;
use super::wordpiece::WordPieceTokenizer;

/// 流式合成时一个片段的最短长度（字符数）。
///
/// 太短的片段（"嗯。""好。"）送进 AR 会因上下文不足很快吐 EOS，韵律差；
/// 而流式的收益是「尽早出声」，切太碎反而两边不讨好。8 个字大约是一个
/// 完整短句，既能开始播音又不至于碎。
const MIN_STREAM_CLAUSE_CHARS: usize = 8;

#[derive(Debug, Clone, Deserialize)]
pub struct VoiceConfig {
    /// 参考音频（相对路径基于模型目录）
    pub ref_audio: String,
    /// 参考音频里说的原话。**给了才走"提示音色"模式**（AR 的 x = 参考文本 + 目标文本，
    /// y = 参考音 codes），和官方 `prompt_text` 一致；不给就是 ref-free（官方也一样，
    /// 见 TTS.py 的 `no_prompt_text -> prompt=None`）。
    #[serde(default)]
    pub ref_text: Option<String>,
    #[serde(default)]
    pub speed: Option<f32>,
}

#[derive(Debug, Default, Deserialize)]
struct VoicesFile {
    #[serde(flatten)]
    voices: HashMap<String, VoiceConfig>,
}

/// 参考音频算出来的东西（同一次进程里复用）。
struct RefFeat {
    /// 参考音 → 语义 codes（AR 的 y）
    prompt: Vec<u32>,
    /// 音色向量（SoVITS 的 ge）
    ge: Tensor,
    /// 参考文本的 phones / BERT；空 = 没给 ref_text，走 ref-free
    text_ids: Vec<u32>,
    bert: Option<Tensor>,
}

pub struct GptSovitsModel {
    dir: PathBuf,
    /// 参考音频特征缓存。参考音频不变就不用重算
    /// （HuBERT + 谱 + ref_enc + eres2net 加起来要几百毫秒）。
    ref_cache: std::sync::Mutex<Option<RefFeat>>,
    frontend: ZhFrontend,
    hubert: Hubert,
    bert: RobertaEncoder,
    tokenizer: WordPieceTokenizer,
    t2s: Text2Semantic,
    sovits: Sovits,
    voices: HashMap<String, VoiceConfig>,
    default_voice: Option<String>,
    device: Device,
}

impl GptSovitsModel {
    pub fn init(model_dir: impl AsRef<Path>, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let dir = model_dir.as_ref().to_path_buf();
        // 没显式指定时：编译了 cuda 且卡可用就用 GPU，否则 CPU
        let device = match device {
            Some(d) => d.clone(),
            None => {
                #[cfg(feature = "cuda")]
                {
                    Device::new_cuda(0).unwrap_or(Device::Cpu)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Device::Cpu
                }
            }
        };
        if device.is_cuda() {
            tracing::info!("GPT-SoVITS 使用 GPU 推理");
        }
        let dtype = dtype.unwrap_or(DType::F32);

        if !dir.join("manifest.json").is_file() {
            bail!(
                "{} 不是 GPT-SoVITS 模型目录：缺少 manifest.json。\n\
                 先用 scripts/gpt_sovits/convert_weights.py 转换权重。",
                dir.display()
            );
        }

        let frontend = ZhFrontend::load(&dir)?;
        let hubert = Hubert::load(&dir, &device, dtype)?;
        let bert = RobertaEncoder::load(&dir, &device, dtype)?;
        let tokenizer = WordPieceTokenizer::load(dir.join("bert_tokenizer/tokenizer.json"))?;
        let t2s = Text2Semantic::load(&dir, &device, dtype)?;
        let sovits = Sovits::load(&dir, &device, dtype)?;

        let (voices, default_voice) = load_voices(&dir, &sovits)?;

        Ok(Self {
            dir,
            ref_cache: std::sync::Mutex::new(None),
            frontend,
            hubert,
            bert,
            tokenizer,
            t2s,
            sovits,
            voices,
            default_voice,
            device,
        })
    }

    pub fn sample_rate(&self) -> usize {
        self.sovits.sampling_rate()
    }

    pub fn voices(&self) -> Vec<&str> {
        self.voices.keys().map(|s| s.as_str()).collect()
    }

    pub fn model_dir(&self) -> &Path {
        &self.dir
    }

    /// 文本 → 波形 `(1, 1, N)`。
    pub fn synthesize(&self, text: &str, voice: Option<&str>, speed: f32) -> Result<Tensor> {
        let cfg = self.resolve_voice(voice)?;
        // 1) 参考音频 → prompt codes + ge（+ 有 ref_text 时的参考文本特征）
        let ref_feat = self.prepare_reference(cfg)?;
        self.synthesize_clause(text, cfg, &ref_feat, speed)
    }

    /// 边合成边吐 **PCM16LE**（桌宠 `POST /tts/stream` 用）。
    ///
    /// # 为什么按句切
    ///
    /// 整段合成时，「第一个音频样本」要等**全部** AR 解码 + 全部声码器跑完才出来；
    /// 长文本下这段等待是几秒。按标点切成片段逐段合成，首段出来就能开始播音——
    /// 首音频延迟从「整段」降到「第一句」。
    ///
    /// 片段之间是自然的句读停顿，不额外插静音。
    ///
    /// 回调拿到的是**这一段**的样本，返回 `false` 可以中止（客户端挂断时用）。
    pub fn synthesize_pcm_stream<F>(
        &self,
        text: &str,
        voice: Option<&str>,
        speed: f32,
        mut emit: F,
    ) -> Result<()>
    where
        F: FnMut(&[i16]) -> bool,
    {
        let cfg = self.resolve_voice(voice)?;
        // 参考特征只算一次，所有片段共用。
        let ref_feat = self.prepare_reference(cfg)?;
        let clauses = self.frontend.split_for_streaming(text, MIN_STREAM_CLAUSE_CHARS);
        if clauses.is_empty() {
            bail!("文本里没有可发音内容：{text:?}");
        }
        for clause in clauses {
            let wave = self.synthesize_clause(&clause, cfg, &ref_feat, speed)?;
            let samples = wave
                .squeeze(0)?
                .squeeze(0)?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;
            let pcm: Vec<i16> = samples
                .iter()
                .map(|s| (s.clamp(-1.0, 1.0) * 32767.0).round() as i16)
                .collect();
            if !emit(&pcm) {
                break;
            }
        }
        Ok(())
    }

    /// 解析音色名（没给就用默认的）。
    fn resolve_voice(&self, voice: Option<&str>) -> Result<&VoiceConfig> {
        let name = voice
            .map(|v| v.to_string())
            .or_else(|| self.default_voice.clone())
            .ok_or_else(|| {
                anyhow::anyhow!("没有可用的音色：请在模型目录里放 voices.json（或 ref.wav）")
            })?;
        self.voices
            .get(&name)
            .ok_or_else(|| anyhow::anyhow!("voices.json 里没有音色 {name:?}"))
    }

    /// 一段文本（不分句）→ 波形 `(1, 1, N)`。参考特征由调用方提供。
    fn synthesize_clause(
        &self,
        text: &str,
        cfg: &VoiceConfig,
        ref_feat: &RefFeat,
        speed: f32,
    ) -> Result<Tensor> {
        // 目标文本前端
        let g2p = self.frontend.g2p(text)?;
        if g2p.phones.is_empty() {
            bail!("文本里没有可发音内容：{text:?}");
        }
        let text_ids = self.frontend.to_ids(&g2p.phones);
        let bert_feat = self
            .bert
            .phone_level_feature(&g2p.norm_text, &self.tokenizer, &g2p.word2ph)?;

        // AR 的输入
        //
        // 官方 `TTS.py`：给了 `prompt_text` 时把**参考文本**的 phones/bert 接在目标文本
        // 前面（`torch.cat([prompt_bert, bert], 1)` + `prompt_phones + phones`），
        // 没给就把 prompt 置空跑 ref-free。少了这一段，AR 看到的是"目标文本 + 别人
        // 的音色 codes"，会立刻吐 EOS（实测 0.4s 就收尾）。
        let (ar_ids, ar_bert) = match &ref_feat.bert {
            Some(ref_bert) => {
                let mut ids = ref_feat.text_ids.clone();
                ids.extend_from_slice(&text_ids);
                (ids, Tensor::cat(&[ref_bert, &bert_feat], 1)?)
            }
            None => (text_ids.clone(), bert_feat),
        };
        let prompt: &[u32] = if ref_feat.bert.is_some() {
            &ref_feat.prompt
        } else {
            &[]
        };

        // AR 出语义 codes
        //
        // 采样偶尔会一直不吐 EOS（一路跑到 1500 token ≈ 60 s 音频），所以按文本长度
        // 设一个上限：约每个 phone 12 帧、最少 48 帧，再兜一个 1500 的硬顶。
        let early_stop = (text_ids.len() * 12 + 48).clamp(48, 1500);
        let gen_opts = GenOptions {
            seed: 42,
            early_stop_num: Some(early_stop),
            ..Default::default()
        };
        let codes = self.t2s.infer(&ar_ids, &ar_bert, prompt, &gen_opts)?;
        if codes.is_empty() {
            bail!("AR 没有生成任何语义 token（文本 {text:?}）");
        }

        // SoVITS 合成（只吃目标文本的 phones）
        let synth = SynthOptions {
            speed: cfg.speed.unwrap_or(speed),
            ..Default::default()
        };
        self.sovits.synthesize(&codes, &text_ids, &ref_feat.ge, &synth)
    }

    /// 参考音频（+ 参考文本）→ prompt codes / ge / 参考文本特征。带缓存。
    fn prepare_reference(&self, cfg: &VoiceConfig) -> Result<RefFeat> {
        if let Some(cached) = self.ref_cache.lock().unwrap().as_ref() {
            return Ok(RefFeat {
                prompt: cached.prompt.clone(),
                ge: cached.ge.clone(),
                text_ids: cached.text_ids.clone(),
                bert: cached.bert.clone(),
            });
        }
        let mut feat = self.prepare_reference_uncached(&cfg.ref_audio)?;
        // 参考文本：官方 `prompt_text`。voices.json 没写时退到参考音旁边的同名 .txt
        //（和 `tools/tts/sovits/ref/ref.wav` + `ref.txt` 一个约定）。
        let ref_text = cfg.ref_text.clone().or_else(|| {
            let mut txt = self.resolve_ref_path(&cfg.ref_audio);
            txt.set_extension("txt");
            std::fs::read_to_string(txt).ok()
        });
        if let Some(text) = ref_text.as_deref().map(str::trim).filter(|t| !t.is_empty()) {
            let g2p = self.frontend.g2p(text)?;
            feat.text_ids = self.frontend.to_ids(&g2p.phones);
            feat.bert = Some(
                self.bert
                    .phone_level_feature(&g2p.norm_text, &self.tokenizer, &g2p.word2ph)?,
            );
            tracing::info!(
                "参考文本 {} 个 phone，走提示音色模式（AR 的 x = 参考文本 + 目标文本）",
                feat.text_ids.len()
            );
        }
        *self.ref_cache.lock().unwrap() = Some(RefFeat {
            prompt: feat.prompt.clone(),
            ge: feat.ge.clone(),
            text_ids: feat.text_ids.clone(),
            bert: feat.bert.clone(),
        });
        Ok(feat)
    }

    /// 参考音频路径（相对路径按模型目录解析）。
    fn resolve_ref_path(&self, ref_audio: &str) -> PathBuf {
        let p = PathBuf::from(ref_audio);
        if p.is_absolute() {
            p
        } else {
            self.dir.join(p)
        }
    }

    fn prepare_reference_uncached(&self, ref_audio: &str) -> Result<RefFeat> {
        let path = self.resolve_ref_path(ref_audio);
        // 音频解码/重采样/STFT 都留在 CPU（GPU 上反而慢），只在进模型前搬到 device
        let cpu = Device::Cpu;
        let (wav, sr) = load_audio(&path, &cpu)?;

        // 16 kHz → HuBERT → 量化；尾部补 0.3 s 静音（官方同样处理）
        let wav16 = resample_simple(&wav, sr as i64, 16_000)?;
        let mut samples: Vec<f32> = wav16.squeeze(0)?.to_vec1()?;
        let seconds = samples.len() as f32 / 16_000.0;
        if !(1.0..=12.0).contains(&seconds) {
            tracing::warn!("参考音频 {seconds:.2} s，官方建议 3~10 s，效果可能不佳");
        }
        samples.extend(std::iter::repeat_n(0.0f32, 16_000 * 3 / 10));

        let feats = self.hubert.forward_pcm(&samples)?;
        let hubert_in = feats.unsqueeze(0)?.transpose(1, 2)?.contiguous()?;
        let prompt = self.sovits.prompt_codes(&hubert_in)?;

        // 32 kHz → 线性幅度谱 → ge
        let wav32 = resample_simple(&wav, sr as i64, self.sovits.sampling_rate() as i64)?;
        let spec = linear_spectrogram(
            &wav32.squeeze(0)?,
            2048,
            self.sovits.hop_length(),
            2048,
            &cpu,
        )?
        .to_device(&self.device)?;
        // v2Pro 的说话人向量模型（eres2net）尚未实现，这里用零向量占位
        let sv = if self.sovits.needs_sv() {
            // 官方对 sv 用的是**不带静音尾**的 16k 参考音频
            let raw16: Vec<f32> = wav16.squeeze(0)?.to_vec1()?;
            Some(self.sovits.sv(&raw16)?)
        } else {
            None
        };
        let ge = self.sovits.ge(&spec, sv.as_ref())?;
        Ok(RefFeat {
            prompt,
            ge,
            text_ids: Vec::new(),
            bert: None,
        })
    }
}

/// wav 用 hound（最稳），其它格式走 symphonia。
fn load_audio(path: &Path, device: &Device) -> Result<(Tensor, usize)> {
    if !path.is_file() {
        bail!("参考音频不存在：{}", path.display());
    }
    let is_wav = path
        .extension()
        .map(|e| e.eq_ignore_ascii_case("wav"))
        .unwrap_or(false);
    if is_wav {
        load_audio_use_hound(path.to_path_buf(), device)
            .with_context(|| format!("读取 {}", path.display()))
    } else {
        let bytes = std::fs::read(path).with_context(|| format!("读取 {}", path.display()))?;
        load_audio_use_symphonia(bytes, device, 1)
            .with_context(|| format!("解码 {}", path.display()))
    }
}

fn load_voices(
    dir: &Path,
    sovits: &Sovits,
) -> Result<(HashMap<String, VoiceConfig>, Option<String>)> {
    let path = dir.join("voices.json");
    let mut voices: HashMap<String, VoiceConfig> = if path.is_file() {
        let file: VoicesFile = serde_json::from_reader(std::io::BufReader::new(
            std::fs::File::open(&path)?,
        ))
        .with_context(|| format!("解析 {}", path.display()))?;
        file.voices
    } else {
        HashMap::new()
    };

    if voices.is_empty() {
        // 退化：模型目录下的 ref.wav
        for name in ["ref.wav", "ref_audio.wav"] {
            if dir.join(name).is_file() {
                tracing::info!("没有 voices.json，使用 {name} 作为参考音频");
                voices.insert(
                    "default".to_string(),
                    VoiceConfig {
                        ref_audio: name.to_string(),
                        ref_text: None,
                        speed: None,
                    },
                );
                break;
            }
        }
    }

    if voices.is_empty() {
        tracing::warn!(
            "{} 下没有 voices.json / ref.wav，synthesize() 会报错（只能跑 prompt_codes 之类的底层接口）",
            dir.display()
        );
        return Ok((voices, None));
    }

    if sovits.is_v2pro() {
        tracing::warn!("当前是 v2Pro：eres2net 说话人向量未实现，音色相似度会低于官方");
    }

    let default = if voices.contains_key("default") {
        Some("default".to_string())
    } else {
        voices.keys().next().cloned()
    };
    Ok((voices, default))
}
