//! IndexTTS 2.5 的端到端拼装。
//!
//! 流程（对照上游 `infer_v2_5.py::infer_generator`）：
//!
//! ```text
//! text tokens ──embed(+lang)──┐
//! speaker w2v ──emo 条件──────┤
//! campplus(192) ─spk_emb_proj─┴─> conds (b,3,1280)  [spk+emo, 0, 0]
//!   └─ GPT 自回归 -> mel codes
//!        └─ semantic_codec.decode(codes) -> 语义特征 (b,2T,1024)
//!             └─ length_regulator ×1.72 -> cond
//!                  └─ cat([prompt_cond, cond]) + prompt mel + style
//!                       └─ CFM 25 步 CFG 采样 -> mel (80)
//!                            └─ bigvgan -> 22.05kHz wav
//! ```

use std::fs::File;
use std::path::Path;

use anyhow::{Result, anyhow, bail};
use candle_core::{D, DType, Device, Tensor};
use candle_nn::{Module, VarBuilder};

use crate::models::bigvgan::{BigVGAN, config::BigVGANConfig};
use crate::models::campplus::CAMPPlus;
use crate::models::common::gguf::Gguf;
use crate::models::common::gguf_lenient;
use crate::models::common::gguf_var_builder::GgufVarBuilder;
use crate::models::index_tts::frontend;
use crate::models::index_tts::{
    config::{GptConfig, IndexTtsConfig},
    emo::EmoConditioner,
    gpt::IndexTtsGpt,
    s2mel::S2Mel,
    tokenizer::{TextTokenizer, lang_to_token},
};
use crate::models::mask_gct::model::RepCodec;
use crate::models::w2v_bert_2_0::{config::W2VBert2_0Config, model::W2VBert2_0Model};

/// 采样参数（默认值和上游 `infer_generator` 一致）
#[derive(Debug, Clone)]
pub struct GenOptions {
    pub max_new_tokens: usize,
    pub do_sample: bool,
    pub top_p: f64,
    pub top_k: usize,
    pub temperature: f64,
    pub repetition_penalty: f64,
    /// beam search 束宽；>1 时走 beam search（上游默认 num_beams=3）
    pub num_beams: usize,
    pub seed: u64,
}

impl Default for GenOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 1500,
            do_sample: true,
            top_p: 0.8,
            top_k: 30,
            temperature: 0.8,
            repetition_penalty: 10.0,
            num_beams: 3,
            seed: 20260101,
        }
    }
}

impl GenOptions {
    /// 快速模式：贪心解码 + 无重复惩罚（RTF ~1.0，适合桌宠实时对话）
    pub fn fast() -> Self {
        Self {
            num_beams: 1,
            do_sample: false,
            repetition_penalty: 1.0,
            ..Default::default()
        }
    }

    /// 质量模式：beam search + 重复惩罚（RTF ~1.8，适合离线合成）
    pub fn quality() -> Self {
        Self::default()
    }
}

/// 极简 xorshift RNG，避免给采样再拉一个依赖，也方便复现
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn sample_token(logits: &[f32], recent: &[u32], opt: &GenOptions, rng: &mut Rng) -> u32 {
    let vocab = logits.len();
    let mut scores: Vec<f64> = logits.iter().map(|v| *v as f64).collect();

    // repetition penalty（HF 语义：正 logit 除、负 logit 乘）
    if opt.repetition_penalty != 1.0 {
        for &t in recent {
            let i = t as usize;
            if i < vocab {
                scores[i] = if scores[i] > 0.0 {
                    scores[i] / opt.repetition_penalty
                } else {
                    scores[i] * opt.repetition_penalty
                };
            }
        }
    }

    if !opt.do_sample {
        let (mut best, mut best_v) = (0usize, f64::NEG_INFINITY);
        for (i, v) in scores.iter().enumerate() {
            if *v > best_v {
                best_v = *v;
                best = i;
            }
        }
        return best as u32;
    }

    let temp = if opt.temperature > 0.0 {
        opt.temperature
    } else {
        1.0
    };
    for v in scores.iter_mut() {
        *v /= temp;
    }

    // top-k：用阈值过滤（k 很小，直接找第 k 大）
    let mut idx: Vec<usize> = (0..vocab).collect();
    idx.sort_unstable_by(|a, b| scores[*b].total_cmp(&scores[*a]));
    if opt.top_k > 0 && opt.top_k < vocab {
        idx.truncate(opt.top_k);
    }

    // top-p
    let max = scores[idx[0]];
    let mut probs: Vec<f64> = idx.iter().map(|i| (scores[*i] - max).exp()).collect();
    let total: f64 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= total;
    }
    if opt.top_p > 0.0 && opt.top_p < 1.0 {
        let mut acc = 0.0;
        let mut keep = probs.len();
        for (i, p) in probs.iter().enumerate() {
            acc += *p;
            if acc >= opt.top_p {
                keep = i + 1;
                break;
            }
        }
        probs.truncate(keep);
        idx.truncate(keep);
        let total: f64 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= total;
        }
    }

    let r = rng.next_f64();
    let mut acc = 0.0;
    for (i, p) in probs.iter().enumerate() {
        acc += *p;
        if r <= acc {
            return idx[i] as u32;
        }
    }
    idx[idx.len() - 1] as u32
}

/// 每个组件开一个独立的 `Gguf`（同一个文件多次打开，读取都是 seek，不会重复占内存）
fn component_vb(
    path: &Path,
    prefix: &str,
    device: &Device,
    dtype: DType,
) -> Result<VarBuilder<'static>> {
    let mut file = File::open(path)?;
    let lc = gguf_lenient::read(&mut file)?;
    let gguf = Gguf::new(lc.into_content(), file, device.clone());
    Ok(VarBuilder::from_backend(
        Box::new(GgufVarBuilder::new(gguf, prefix)),
        dtype,
        device.clone(),
    ))
}

fn read_config(path: &Path) -> Result<IndexTtsConfig> {
    let mut file = File::open(path)?;
    let lc = gguf_lenient::read(&mut file)?;
    let bytes = gguf_lenient::embedded_file(&mut file, &lc, "config.yaml")?
        .ok_or_else(|| anyhow!("GGUF 里没有 config.yaml"))?;
    IndexTtsConfig::from_yaml(&bytes)
}

fn read_embedded_json<T: serde::de::DeserializeOwned>(path: &Path, name: &str) -> Result<T> {
    let mut file = File::open(path)?;
    let lc = gguf_lenient::read(&mut file)?;
    let bytes = gguf_lenient::embedded_file(&mut file, &lc, name)?
        .ok_or_else(|| anyhow!("GGUF 里没有 {name}"))?;
    Ok(serde_json::from_slice(&bytes)?)
}

/// 上游 `remove_long_silence`：截断到 stop token；若静音 code 总数超过 `max_consecutive`，
/// 则把每段连续静音压到最多 10 个（停顿太长听起来会"断"）
pub fn remove_long_silence(
    codes: Vec<u32>,
    stop_token: u32,
    silent_token: u32,
    max_consecutive: usize,
) -> Vec<u32> {
    let len = codes
        .iter()
        .position(|c| *c == stop_token)
        .unwrap_or(codes.len());
    let count = codes.iter().filter(|c| **c == silent_token).count();
    if count > max_consecutive {
        let mut out = Vec::with_capacity(len);
        let mut n = 0usize;
        for &c in &codes[..len] {
            if c != silent_token {
                out.push(c);
                n = 0;
            } else if n < 10 {
                out.push(c);
                n += 1;
            }
        }
        out
    } else {
        codes[..len].to_vec()
    }
}

pub struct IndexTtsPipeline {
    pub config: GptConfig,
    pub sample_rate: usize,
    pub device: Device,
    pub gpt_dtype: DType,
    pub cfm_dtype: DType,
    pub bigvgan_dtype: DType,
    pub gpt: IndexTtsGpt,
    pub emo: EmoConditioner,
    pub s2mel: S2Mel,
    pub codec: RepCodec,
    pub bigvgan: BigVGAN,
    pub w2v_bert: W2VBert2_0Model,
    pub campplus: CAMPPlus,
    pub wav_stats_mean: Tensor,
    pub wav_stats_std: Tensor,
}

impl IndexTtsPipeline {
    pub fn from_gguf(path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        Self::from_gguf_with_dtype(path, device, DType::F32)
    }

    /// 指定权重 dtype（CUDA 上显存吃紧时可以传 `DType::F16`）
    pub fn from_gguf_with_dtype(
        path: impl AsRef<Path>,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        Self::from_gguf_split_dtype(path, device, dtype, dtype)
    }

    /// GPT、CFM、BigVGAN 分别用不同精度
    pub fn from_gguf_split_dtype(
        path: impl AsRef<Path>,
        device: &Device,
        gpt_dtype: DType,
        cfm_dtype: DType,
    ) -> Result<Self> {
        Self::from_gguf_full_split(path, device, gpt_dtype, cfm_dtype, cfm_dtype)
    }

    /// GPT、CFM、BigVGAN 各自的精度（BigVGAN 用 F16 通常无损且能翻倍提速）
    pub fn from_gguf_full_split(
        path: impl AsRef<Path>,
        device: &Device,
        gpt_dtype: DType,
        cfm_dtype: DType,
        bigvgan_dtype: DType,
    ) -> Result<Self> {
        let path = path.as_ref();
        let cfg = read_config(path)?;

        // ---- GPT（含 emo 条件，同一个 VarBuilder）
        let gpt_vb = component_vb(path, "gpt", device, gpt_dtype)?;
        let gpt = IndexTtsGpt::new(&gpt_vb, &cfg.gpt)?;
        let emo_cfg = cfg
            .gpt
            .emo_condition_module
            .clone()
            .ok_or_else(|| anyhow!("config.yaml 缺少 emo_condition_module"))?;
        let emo = EmoConditioner::new(
            &gpt_vb,
            1024,
            emo_cfg.output_size,
            emo_cfg.attention_heads,
            emo_cfg.linear_units,
            emo_cfg.num_blocks,
            1024,
            emo_cfg.perceiver_mult,
            cfg.gpt.model_dim,
        )?;

        // ---- s2mel（隐藏维/层数来自 config.yaml 的 DiT 段，这里按 2.5 的定值传）
        let s2mel_vb = component_vb(path, "s2mel", device, cfm_dtype)?;
        let s2mel = S2Mel::new(&s2mel_vb, 80, 512, 8, 13, 512, 1024, 512, 4, 8, 1280)?;

        // ---- semantic codec
        let mut codec_cfg = cfg
            .semantic_codec
            .clone()
            .ok_or_else(|| anyhow!("config.yaml 缺少 semantic_codec"))?;
        codec_cfg.downsample_scale = 2; // 有 down/up 卷积
        codec_cfg.num_quantizers = 1;
        let codec_vb = component_vb(path, "semantic_codec", device, cfm_dtype)?;
        let codec = RepCodec::new(codec_vb, &codec_cfg)?;

        // ---- bigvgan
        let bv_cfg: BigVGANConfig = read_embedded_json(path, "bigvgan/config.json")?;
        let bv_vb = component_vb(path, "bigvgan", device, bigvgan_dtype)?;
        let bigvgan = BigVGAN::new(bv_vb, &bv_cfg)?;

        // ---- w2v-bert（语义特征，取第 17 层 hidden）
        let w2v_cfg: W2VBert2_0Config = read_embedded_json(path, "w2v-bert-2.0/config.json")?;
        let w2v_vb = component_vb(path, "wav2vec2bert", device, gpt_dtype)?;
        let w2v_bert = W2VBert2_0Model::new(w2v_vb, &w2v_cfg)?;

        // ---- campplus（说话人 192 维）
        let cp_vb = component_vb(path, "campplus/speaker_encoder", device, gpt_dtype)?;
        let campplus = CAMPPlus::new(cp_vb, 80, 192, 32, 4, 128)?;

        // ---- w2v-bert 输出归一化统计量
        let stats_vb = component_vb(path, "wav2vec2bert_stats", device, gpt_dtype)?;
        let wav_stats_mean = stats_vb.get(1024, "mean")?.to_dtype(gpt_dtype)?;
        let wav_stats_var = stats_vb.get(1024, "var")?.to_dtype(gpt_dtype)?;
        let wav_stats_std = (wav_stats_var + 1e-7)?.sqrt()?;

        Ok(Self {
            config: cfg.gpt,
            sample_rate: bv_cfg.sampling_rate,
            device: device.clone(),
            gpt_dtype,
            cfm_dtype,
            bigvgan_dtype,
            gpt,
            emo,
            s2mel,
            codec,
            bigvgan,
            w2v_bert,
            campplus,
            wav_stats_mean,
            wav_stats_std,
        })
    }

    /// GPT 侧的 dtype（用于跨精度数据传递时转换）
    fn gpt_dtype(&self) -> DType {
        self.gpt_dtype
    }

    /// CFM 侧的 dtype
    fn cfm_dtype(&self) -> DType {
        self.cfm_dtype
    }

    /// BigVGAN 的 dtype
    fn bigvgan_dtype(&self) -> DType {
        self.bigvgan_dtype
    }

    /// 加载 GGUF 内嵌的 tiktoken 词表
    pub fn text_tokenizer(&self, path: impl AsRef<Path>) -> Result<TextTokenizer> {
        let mut file = File::open(path.as_ref())?;
        let lc = gguf_lenient::read(&mut file)?;
        let bytes = gguf_lenient::embedded_file(
            &mut file,
            &lc,
            "multilingual_zh_ja_yue_char_del.tiktoken",
        )?
        .ok_or_else(|| anyhow!("GGUF 里没有 .tiktoken 词表"))?;
        TextTokenizer::from_tiktoken_bytes(&bytes)
    }

    /// 文本 -> (token ids, lang_embedding 下标)，和上游一样加 `<|lang|> ` 前缀
    pub fn encode_text(
        &self,
        tok: &TextTokenizer,
        lang: &str,
        text: &str,
    ) -> Result<(Tensor, u32)> {
        let prefix = format!("<|{}|> ", lang.to_lowercase());
        let ids = tok.encode(&format!("{prefix}{text}"))?;
        let n = ids.len();
        let ids = Tensor::from_vec(ids, (1, n), &self.device)?;
        Ok((ids, lang_to_token(lang)))
    }

    /// 说话人 + 情绪 -> 3 个条件 latent：`[spk+emo, 0, 0]`
    pub fn cond_latents(
        &self,
        spk_w2v: &Tensor,
        emo_w2v: &Tensor,
        campplus_emb: &Tensor,
        emo_alpha: f64,
    ) -> Result<Tensor> {
        let base = self.emo.forward(spk_w2v)?;
        let emo_vec = self.emo.forward(emo_w2v)?;
        let mixed = (&base + ((&emo_vec - &base)? * emo_alpha)?)?;
        let spk = self.gpt.spk_emb_proj.forward(campplus_emb)?; // (b, model_dim)
        let latent = (spk + mixed)?;
        let (b, dim) = latent.dims2()?;
        let zeros = Tensor::zeros((b, 2, dim), latent.dtype(), latent.device())?;
        Ok(Tensor::cat(&[latent.unsqueeze(1)?, zeros], 1)?)
    }

    /// 文本 token（不含 start/stop）-> 自回归生成 mel codes (b, T)
    pub fn generate(
        &mut self,
        text_ids: &Tensor,
        lang_id: u32,
        conds: &Tensor,
        opt: &GenOptions,
    ) -> Result<Tensor> {
        let rows = text_ids.to_vec2::<u32>()?;
        if rows.len() != 1 {
            bail!("目前只支持 batch=1，收到 {}", rows.len());
        }
        let mut ids = vec![self.config.start_text_token as u32];
        ids.extend(rows[0].iter().copied());
        ids.push(self.config.stop_text_token as u32);

        let device = text_ids.device();
        let text_len = ids.len();
        let ids = Tensor::from_vec(ids, (1, text_len), device)?;
        let text_emb = self.gpt.embed_text_with_lang(&ids, lang_id)?; // (1,L,dim)

        let cond_len = conds.dim(1)?;
        let prefix_len = cond_len + text_len;
        let prefix = Tensor::cat(&[conds, &text_emb], 1)?;
        let start = Tensor::from_vec(vec![self.config.start_mel_token as u32], (1, 1), device)?;
        // 注意：mel 位置编码是「相对 mel 段」的（上游 start token 用 0，之后 1,2,...），
        // 不是整条序列的绝对位置，这里用 prefix_len 会整体偏移，音色和音质都会坏。
        let start_emb = self.gpt.embed_mel(&start, 0)?;
        let embeds = Tensor::cat(&[&prefix, &start_emb], 1)?;

        // beam search（上游 num_beams=3）：束宽 > 1 时走这条
        if opt.num_beams > 1 {
            let codes = self.generate_beam(&embeds, opt)?;
            let codes = remove_long_silence(codes, self.config.stop_mel_token as u32, 52, 30);
            let n = codes.len();
            return Ok(Tensor::from_vec(codes, (1, n), device)?.to_dtype(DType::U32)?);
        }

        let mut hidden = self.gpt.forward(&embeds, 0)?;
        let mut codes: Vec<u32> = Vec::with_capacity(opt.max_new_tokens);
        let mut rng = Rng::new(opt.seed);
        let stop = self.config.stop_mel_token as u32;
        let start_mel = self.config.start_mel_token as u32;

        for step in 0..opt.max_new_tokens {
            let seq = hidden.dim(1)?;
            let last = hidden.narrow(1, seq - 1, 1)?;
            let logits = self.gpt.mel_logits(&last)?; // (1,1,V)
            let mut v = logits.to_vec3::<f32>()?[0][0].clone();
            if (start_mel as usize) < v.len() {
                v[start_mel as usize] = f32::NEG_INFINITY; // 起始 token 不能再被采样
            }
            let tok = sample_token(&v, &codes, opt, &mut rng);
            if tok == stop {
                break;
            }
            codes.push(tok);
            let pos = 1 + step;
            let id = Tensor::from_vec(vec![tok], (1, 1), device)?;
            let emb = self.gpt.embed_mel(&id, pos)?;
            hidden = self.gpt.forward(&emb, pos)?;
        }

        if codes.is_empty() {
            bail!("GPT 没有生成任何 mel code");
        }
        // 对齐上游 `remove_long_silence`：截断到 stop token，并把连续静音 code 压到最多 10 个
        let codes = remove_long_silence(codes, self.config.stop_mel_token as u32, 52, 30);
        let n = codes.len();
        Ok(Tensor::from_vec(codes, (1, n), device)?.to_dtype(DType::U32)?)
    }

    /// beam search 主体：把 beams 当 batch 维一起跑，每步重排 KV cache
    fn generate_beam(&mut self, embeds: &Tensor, opt: &GenOptions) -> Result<Vec<u32>> {
        let beams = opt.num_beams;
        let device = embeds.device();
        let stop = self.config.stop_mel_token as u32;
        let start_mel = self.config.start_mel_token as u32;

        // 预填充：所有 beam 共享同一个前缀（等于把 embeds 复制 beams 份）
        let embeds = embeds.repeat((beams, 1, 1))?;
        let mut hidden = self.gpt.forward(&embeds, 0)?; // (beams, L, dim)

        // 每个 beam 的累计 log-prob；beam 0 起点为 0，其余为 -inf（避免第一步选出重复候选）
        let mut scores = vec![f64::NEG_INFINITY; beams];
        scores[0] = 0.0;
        let mut seqs: Vec<Vec<u32>> = vec![Vec::new(); beams];
        // 已结束的 beam：()：记录 (score, codes)
        let mut finished: Vec<(f64, Vec<u32>)> = Vec::new();

        for step in 0..opt.max_new_tokens {
            let seq = hidden.dim(1)?;
            let last = hidden.narrow(1, seq - 1, 1)?; // (beams, 1, dim)
            let logits = self.gpt.mel_logits(&last)?; // (beams, 1, V)
            let logits = logits.squeeze(1)?.to_dtype(candle_core::DType::F32)?;
            let raw = logits.to_vec2::<f32>()?; // (beams, V)
            let vocab = raw.first().map(|r| r.len()).unwrap_or(0);

            // 每个 beam 单独做 repetition penalty（HF 语义：正 logit 除、负 logit 乘）再 log_softmax
            let logp: Vec<Vec<f32>> = raw
                .iter()
                .enumerate()
                .map(|(b, row)| {
                    let mut r = row.clone();
                    if (start_mel as usize) < r.len() {
                        r[start_mel as usize] = f32::NEG_INFINITY;
                    }
                    if opt.repetition_penalty != 1.0 {
                        for prev in seqs[b].iter() {
                            let i = *prev as usize;
                            if i < r.len() {
                                r[i] = if r[i] > 0.0 {
                                    r[i] / opt.repetition_penalty as f32
                                } else {
                                    r[i] * opt.repetition_penalty as f32
                                };
                            }
                        }
                    }
                    let max = r.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let sum: f32 = r.iter().map(|v| (v - max).exp()).sum();
                    r.iter().map(|v| (v - max) - sum.ln()).collect()
                })
                .collect();

            // 候选：(score, 来自 beam, token)
            let mut cands: Vec<(f64, usize, u32)> = Vec::with_capacity(beams * 64);
            for b in 0..beams {
                if scores[b] == f64::NEG_INFINITY {
                    continue;
                }
                for t in 0..vocab {
                    let tok = t as u32;
                    if tok == start_mel {
                        continue;
                    }
                    let s = scores[b] + logp[b][t] as f64;
                    cands.push((s, b, tok));
                }
            }
            if cands.is_empty() {
                break;
            }
            cands.sort_by(|a, b| b.0.total_cmp(&a.0));

            // 取前 beams 个；命中的 beam 若生成 stop token 则记入 finished
            let mut new_scores = vec![f64::NEG_INFINITY; beams];
            let mut new_seqs: Vec<Vec<u32>> = vec![Vec::new(); beams];
            let mut parents = vec![0usize; beams];
            let mut picked = 0usize;
            for (s, b, tok) in cands {
                if picked >= beams {
                    break;
                }
                if tok == stop {
                    // 该 beam 到达终点，记录并继续找下一个候选
                    let mut codes = seqs[b].clone();
                    codes.push(tok);
                    finished.push((s, codes));
                    continue;
                }
                let mut codes = seqs[b].clone();
                codes.push(tok);
                new_scores[picked] = s;
                new_seqs[picked] = codes;
                parents[picked] = b;
                picked += 1;
            }
            if picked == 0 {
                break;
            }
            (scores, seqs) = (new_scores, new_seqs);

            // 重排 KV cache，再喂新 token
            let idx = Tensor::from_vec(parents.iter().map(|p| *p as u32).collect(), beams, device)?;
            self.gpt.reorder_kv_cache(&idx)?;
            let mut embs = Vec::with_capacity(beams);
            for b in 0..beams {
                let last_tok = seqs[b].last().copied().unwrap_or(start_mel);
                let id = Tensor::from_vec(vec![last_tok], (1, 1), device)?;
                embs.push(self.gpt.embed_mel(&id, 1 + step)?);
            }
            // 每个 beam 喂的是 (1,1,dim)，拼成 (beams,1,dim)
            let mut stacked = embs[0].clone();
            for e in embs.iter().skip(1) {
                stacked = Tensor::cat(&[stacked, e.clone()], 0)?;
            }
            hidden = self.gpt.forward(&stacked, 1 + step)?;

            // 所有 beam 都没东西可选了就停
            if scores.iter().all(|s| *s == f64::NEG_INFINITY) {
                break;
            }
        }

        // 选最优：优先已结束的 beam，否则取当前最优未结束 beam
        if !finished.is_empty() {
            finished.sort_by(|a, b| b.0.total_cmp(&a.0));
            let (_, codes) = finished.remove(0);
            Ok(codes)
        } else {
            let best = scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .unwrap_or(0);
            Ok(seqs[best].clone())
        }
    }

    /// 语义码 -> mel -> wav（`prompt_mel` 与 `prompt_cond` 的长度必须一致）
    pub fn codes_to_wav(
        &mut self,
        codes: &Tensor,
        prompt_mel: &Tensor,
        prompt_cond: &Tensor,
        style: &Tensor,
        diffusion_steps: usize,
        cfg_rate: f64,
    ) -> Result<Tensor> {
        // 统一转成 CFM 的精度（GPT F32 -> CFM F16 时需要这一步）
        let cfm_dtype = self.cfm_dtype;
        let prompt_mel = prompt_mel.to_dtype(cfm_dtype)?;
        let prompt_cond = prompt_cond.to_dtype(cfm_dtype)?;
        let style = style.to_dtype(cfm_dtype)?;
        let s_infer = self.codec.decode_codes(codes)?.to_dtype(cfm_dtype)?; // (1, 2T, 1024)
        let frames = s_infer.dim(1)?;
        let target = (frames as f64 * 1.72) as usize;
        let cond = self.s2mel.length_regulator.forward(&s_infer, target)?;
        let cond = cond.to_dtype(cfm_dtype)?;

        let prompt_frames = prompt_cond.dim(1)?;
        let cat_condition = Tensor::cat(&[&prompt_cond, &cond], 1)?;
        let t_cfm = std::time::Instant::now();
        let mel = self.s2mel.cfm.solve_euler(
            &cat_condition,
            &prompt_mel,
            &style,
            diffusion_steps,
            cfg_rate,
            1.0,
        )?;
        eprintln!(
            "[perf] CFM({diffusion_steps}步) = {} ms",
            t_cfm.elapsed().as_millis()
        );
        // BigVGAN 的权重可能是 F16，CFM 输出的 mel 也要跟着转
        let mel = mel.to_dtype(self.bigvgan_dtype())?;
        // 去掉 prompt 区间，只留生成部分
        let total = prompt_frames + target;
        let mel = mel.narrow(
            D::Minus1,
            prompt_mel.dim(D::Minus1)?,
            total - prompt_mel.dim(D::Minus1)?,
        )?;
        let t_bv = std::time::Instant::now();
        let wav = self.bigvgan.forward(&mel)?;
        eprintln!("[perf] BigVGAN = {} ms", t_bv.elapsed().as_millis());
        Ok(wav)
    }

    /// 参考音频 + 文本 -> wav 采样点（16k/22.05k 重采样、w2v-bert、campplus、ref_mel 全在这一步）
    /// 返回 (wav, 各阶段耗时毫秒)
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_profiled(
        &mut self,
        tok: &TextTokenizer,
        ref_wav: &str,
        text: &str,
        lang: &str,
        opt: &GenOptions,
        diffusion_steps: usize,
        cfg_rate: f64,
    ) -> Result<(Vec<f32>, Vec<(&'static str, u128)>)> {
        let mut timings = Vec::new();
        let t = std::time::Instant::now();

        let (audio, sr) = frontend::load_wav(ref_wav)?;
        let max_samples = 15 * sr;
        let audio = if audio.len() > max_samples {
            audio[..max_samples].to_vec()
        } else {
            audio
        };
        let audio_16k = frontend::resample(&audio, sr, 16000);
        let audio_22k = frontend::resample(&audio, sr, 22050);
        timings.push(("frontend(resample)", t.elapsed().as_millis()));
        let t = std::time::Instant::now();

        let feats = frontend::seamless_features(&audio_16k, &self.device)?;
        let out = self.w2v_bert.forward(&feats, None, Some(17), true)?;
        let hidden = out
            .specify_layer_id_hidden_state
            .unwrap_or(out.last_hidden_state);
        let cond_feat = hidden
            .broadcast_sub(&self.wav_stats_mean)?
            .broadcast_div(&self.wav_stats_std)?;
        timings.push(("w2v_bert", t.elapsed().as_millis()));
        let t = std::time::Instant::now();

        let fbank = frontend::campplus_fbank(&audio_16k, &self.device)?;
        let style = self.campplus.forward(&fbank)?;
        let style = style.to_dtype(self.gpt_dtype())?; // GPT 和 CFM 可能精度不同
        timings.push(("campplus", t.elapsed().as_millis()));
        let t = std::time::Instant::now();

        let ref_mel = frontend::mel_spectrogram_22050(&audio_22k, &self.device)?;
        let ref_mel = ref_mel.to_dtype(self.cfm_dtype())?;
        timings.push(("mel_frontend", t.elapsed().as_millis()));
        let t = std::time::Instant::now();

        let conds = self.cond_latents(&cond_feat, &cond_feat, &style, 1.0)?;
        let prompt_len = ref_mel.dim(D::Minus1)?;
        // length_regulator 是 s2mel 的一部分，权重精度可能是 F16；输入要先转
        let cond_feat_cfm = cond_feat.to_dtype(self.cfm_dtype)?;
        let prompt_cond = self
            .s2mel
            .length_regulator
            .forward(&cond_feat_cfm, prompt_len)?;
        let prompt_cond = prompt_cond.to_dtype(self.cfm_dtype())?;
        timings.push(("cond_latents+LR", t.elapsed().as_millis()));
        let t = std::time::Instant::now();

        let (text_ids, lang_id) = self.encode_text(tok, lang, text)?;
        let codes = self.generate(&text_ids, lang_id, &conds, opt)?;
        timings.push(("gpt_generate", t.elapsed().as_millis()));
        let t = std::time::Instant::now();

        let wav = self.codes_to_wav(
            &codes,
            &ref_mel,
            &prompt_cond,
            &style,
            diffusion_steps,
            cfg_rate,
        )?;
        timings.push(("cfm+bigvgan", t.elapsed().as_millis()));

        Ok((wav.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?, timings))
    }

    /// 参考音频 + 文本 -> wav 采样点（16k/22.05k 重采样、w2v-bert、campplus、ref_mel 全在这一步）
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize(
        &mut self,
        tok: &TextTokenizer,
        ref_wav: &str,
        text: &str,
        lang: &str,
        opt: &GenOptions,
        diffusion_steps: usize,
        cfg_rate: f64,
    ) -> Result<Vec<f32>> {
        let (wav, _) =
            self.synthesize_profiled(tok, ref_wav, text, lang, opt, diffusion_steps, cfg_rate)?;
        Ok(wav)
    }
}
