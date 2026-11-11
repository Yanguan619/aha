//! GPT-SoVITS 的 AR 部分：Text2Semantic（`s1*.ckpt`）。
//!
//! 严格对齐 `AR/models/t2s_model.py::Text2SemanticDecoder`：
//!
//! ```text
//! 文本: ar_text_embedding(phones) + bert_proj(bert^T) + sine_pe
//! 提示: ar_audio_embedding(prompt_codes) + sine_pe
//! 拼成 [text; prompt] 做一次全量 attention（mask 很特殊，见下）
//!   → 取最后一行的 logits → 采样 → 逐步用 cache 解码，最多 1500 步
//! ```
//!
//! **mask 结构**（`infer_panel_naive` 原文）：
//!
//! |            | 文本列 | 音频列 |
//! |------------|--------|--------|
//! | 文本行     | 可见   | **屏蔽** |
//! | 音频行     | 可见   | 因果   |
//!
//! 也就是说文本 token 看不到参考音频，参考音频能看到全部文本。
//!
//! 每层是 post-norm：`x = LN1(x + attn); x = LN2(x + mlp(x))`，MLP 用 ReLU。

use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder, embedding, linear, linear_no_bias};
use serde::Deserialize;

use crate::models::common::sample::simple_sample;

#[derive(Debug, Clone, Deserialize)]
pub struct T2SModelConfig {
    pub vocab_size: usize,
    pub phoneme_vocab_size: usize,
    pub embedding_dim: usize,
    pub hidden_dim: usize,
    pub head: usize,
    pub n_layer: usize,
    #[serde(default)]
    pub linear_units: usize,
    #[serde(default)]
    pub dropout: f64,
    pub eos: usize,
}

#[derive(Debug, Deserialize)]
struct T2SSection {
    model: T2SInner,
}

#[derive(Debug, Deserialize)]
struct T2SInner {
    vocab_size: usize,
    phoneme_vocab_size: usize,
    embedding_dim: usize,
    hidden_dim: usize,
    head: usize,
    n_layer: usize,
    #[serde(default)]
    linear_units: usize,
    #[serde(default)]
    dropout: f64,
    #[serde(rename = "EOS")]
    eos: usize,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    t2s_config: T2SSection,
}

impl T2SModelConfig {
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        let path = dir.as_ref().join("manifest.json");
        let manifest: Manifest = serde_json::from_reader(std::io::BufReader::new(
            std::fs::File::open(&path).with_context(|| format!("打开 {}", path.display()))?,
        ))
        .with_context(|| format!("解析 {}", path.display()))?;
        let m = manifest.t2s_config.model;
        Ok(Self {
            vocab_size: m.vocab_size,
            phoneme_vocab_size: m.phoneme_vocab_size,
            embedding_dim: m.embedding_dim,
            hidden_dim: m.hidden_dim,
            head: m.head,
            n_layer: m.n_layer,
            linear_units: m.linear_units,
            dropout: m.dropout,
            eos: m.eos,
        })
    }
}

/// 生成参数（默认值与 `TTS.py` 一致）。
#[derive(Debug, Clone)]
pub struct GenOptions {
    pub greedy: bool,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub early_stop_num: Option<usize>,
    pub max_new_tokens: usize,
    pub seed: u64,
}

impl Default for GenOptions {
    fn default() -> Self {
        Self {
            greedy: false,
            temperature: 1.0,
            top_k: 15,
            top_p: 1.0,
            repetition_penalty: 1.35,
            early_stop_num: None,
            max_new_tokens: 1500,
            seed: 42,
        }
    }
}

struct Block {
    qkv_w: Tensor,
    qkv_b: Tensor,
    out: Linear,
    f1: Linear,
    f2: Linear,
    n1w: Tensor,
    n1b: Tensor,
    n2w: Tensor,
    n2b: Tensor,
    eps: f64,
    heads: usize,
    head_dim: usize,
}

impl Block {
    fn new(vb: VarBuilder, hidden: usize, heads: usize, eps: f64) -> Result<Self> {
        Ok(Self {
            qkv_w: vb.get((hidden * 3, hidden), "self_attn.in_proj_weight")?,
            qkv_b: vb.get(hidden * 3, "self_attn.in_proj_bias")?,
            out: linear(hidden, hidden, vb.pp("self_attn.out_proj"))?,
            f1: linear(hidden, hidden * 4, vb.pp("linear1"))?,
            f2: linear(hidden * 4, hidden, vb.pp("linear2"))?,
            n1w: vb.get(hidden, "norm1.weight")?,
            n1b: vb.get(hidden, "norm1.bias")?,
            n2w: vb.get(hidden, "norm2.weight")?,
            n2b: vb.get(hidden, "norm2.bias")?,
            eps,
            heads,
            head_dim: hidden / heads,
        })
    }

    fn layer_norm(&self, x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
        let mean = x.mean_keepdim(candle_core::D::Minus1)?;
        let centered = x.broadcast_sub(&mean)?;
        let var = centered.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let norm = (var + self.eps)?.sqrt()?.recip()?;
        let out = centered.broadcast_mul(&norm)?;
        Ok(out.broadcast_mul(w)?.broadcast_add(b)?)
    }

    /// `cache = None` 时处理整段（prompt 阶段）；`Some((k, v))` 时增量解码一个 token。
    fn forward(
        &self,
        x: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, t, hidden) = x.dims3()?;
        // 融合 QKV：candle 的 3D×2D matmul 有形状限制，这里按 b=1 展平再算
        let flat = x.reshape((b * t, hidden))?;
        let qkv = flat.matmul(&self.qkv_w.t()?)?.broadcast_add(&self.qkv_b)?;
        let qkv = qkv.reshape((b, t, hidden * 3))?;
        let q = qkv.narrow(2, 0, hidden)?;
        let k = qkv.narrow(2, hidden, hidden)?;
        let v = qkv.narrow(2, hidden * 2, hidden)?;

        let (k_cache, v_cache) = match cache {
            Some((kc, vc)) => (
                Tensor::cat(&[kc, &k], 1)?,
                Tensor::cat(&[vc, &v], 1)?,
            ),
            None => (k, v),
        };

        let kv_len = k_cache.dim(1)?;
        let q = q.reshape((b, t, self.heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let k = k_cache.reshape((b, kv_len, self.heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let v = v_cache.reshape((b, kv_len, self.heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;

        let mut scores = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        scores = (scores / (self.head_dim as f64).sqrt())?;
        if let Some(mask) = mask {
            scores = scores.broadcast_add(mask)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let attn = probs.matmul(&v)?;
        let attn = attn
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, hidden))?;
        let attn = self.out.forward(&attn)?;

        let x = self.layer_norm(&(x + attn)?, &self.n1w, &self.n1b)?;
        let mlp = self.f2.forward(&self.f1.forward(&x)?.relu()?)?;
        let x = self.layer_norm(&(&x + mlp)?, &self.n2w, &self.n2b)?;
        Ok((x, k_cache, v_cache))
    }
}

pub struct Text2Semantic {
    bert_proj: Linear,
    text_emb: Embedding,
    audio_emb: Embedding,
    alpha_text: Tensor,
    alpha_audio: Tensor,
    blocks: Vec<Block>,
    predict: Linear,
    device: Device,
    pub cfg: T2SModelConfig,
}

impl Text2Semantic {
    pub fn load(dir: impl AsRef<Path>, device: &Device, dtype: DType) -> Result<Self> {
        let dir = dir.as_ref();
        let cfg = T2SModelConfig::load(dir)?;
        let tensors = candle_core::safetensors::load(dir.join("gpt.safetensors"), device)?;
        let vb = VarBuilder::from_tensors(tensors, dtype, device);

        let hidden = cfg.hidden_dim;
        let mut blocks = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            blocks.push(Block::new(vb.pp("h.layers").pp(i), hidden, cfg.head, 1e-5)?);
        }

        let model = Self {
            bert_proj: linear(1024, cfg.embedding_dim, vb.pp("bert_proj"))?,
            text_emb: embedding(
                cfg.phoneme_vocab_size,
                cfg.embedding_dim,
                vb.pp("ar_text_embedding.word_embeddings"),
            )?,
            audio_emb: embedding(
                cfg.vocab_size,
                cfg.embedding_dim,
                vb.pp("ar_audio_embedding.word_embeddings"),
            )?,
            alpha_text: vb.get(1, "ar_text_position.alpha")?,
            alpha_audio: vb.get(1, "ar_audio_position.alpha")?,
            blocks,
            predict: linear_no_bias(cfg.hidden_dim, cfg.vocab_size, vb.pp("ar_predict_layer"))?,
            device: device.clone(),
            cfg,
        };
        Ok(model)
    }

    /// 正弦位置编码（`SinePositionalEmbedding`，`scale=False`）：(1, len, dim)
    fn sine_pe(&self, len: usize, dtype: DType) -> Result<Tensor> {
        let dim = self.cfg.embedding_dim;
        let half = dim / 2;
        let log_scale = 10_000f64.ln() / dim as f64;
        let mut values = vec![0f32; len * dim];
        for pos in 0..len {
            for i in 0..half {
                // 注意：div_term 用的是 arange(0, dim, 2)，即 2*i（不是 i）
                let div = (-(2.0 * i as f64) * log_scale).exp();
                let angle = pos as f64 * div;
                values[pos * dim + 2 * i] = angle.sin() as f32;
                values[pos * dim + 2 * i + 1] = angle.cos() as f32;
            }
        }
        Ok(Tensor::from_vec(values, (1, len, dim), &self.device)?.to_dtype(dtype)?)
    }

    fn add_pe(&self, x: &Tensor, alpha: &Tensor, pos: usize) -> Result<Tensor> {
        let pe = self.sine_pe(pos + 1, x.dtype())?.narrow(1, pos, 1)?;
        Ok((x + pe.broadcast_mul(alpha)?)?)
    }

    /// 生成语义 token（返回**不含**提示部分的生成结果）。

    /// 诊断：返回 (拼好进 transformer 的输入, 第一步 logits)
    pub fn probe_first_step(
        &self,
        text_ids: &[u32],
        bert_feature: &Tensor,
        prompt_codes: &[u32],
    ) -> Result<(Tensor, Tensor)> {
        let x_len = text_ids.len();
        let y_len = prompt_codes.len();

        let ids = Tensor::from_vec(text_ids.to_vec(), (1, x_len), &self.device)?;
        let text = self.text_emb.forward(&ids)?;
        let bert = self.bert_proj.forward(&bert_feature.t()?.unsqueeze(0)?)?;
        let text = (&text + bert)?;
        let text = (&text + self.sine_pe(x_len, text.dtype())?.broadcast_mul(&self.alpha_text)?)?;

        let (xy, mask) = if y_len > 0 {
            let pids = Tensor::from_vec(prompt_codes.to_vec(), (1, y_len), &self.device)?;
            let pemb = self.audio_emb.forward(&pids)?;
            let prompt =
                (&pemb + self.sine_pe(y_len, pemb.dtype())?.broadcast_mul(&self.alpha_audio)?)?;
            let xy = Tensor::cat(&[&text, &prompt], 1)?;
            let mask = self.build_prompt_mask(x_len, y_len)?;
            (xy, Some(mask))
        } else {
            (text, None)
        };

        let mut hidden = xy.clone();
        for block in &self.blocks {
            let (out, _, _) = block.forward(&hidden, None, mask.as_ref())?;
            hidden = out;
        }
        let last = hidden.narrow(1, hidden.dim(1)? - 1, 1)?;
        let logits = self.predict.forward(&last.squeeze(1)?)?.squeeze(0)?;
        Ok((xy, logits))
    }

    pub fn infer(
        &self,
        text_ids: &[u32],
        bert_feature: &Tensor,
        prompt_codes: &[u32],
        opts: &GenOptions,
    ) -> Result<Vec<u32>> {
        let x_len = text_ids.len();
        let y_len = prompt_codes.len();

        // 文本: embedding + bert_proj + pe
        let ids = Tensor::from_vec(text_ids.to_vec(), (1, x_len), &self.device)?;
        let text = self.text_emb.forward(&ids)?; // (1, S, D)
        let bert = self.bert_proj.forward(&bert_feature.t()?.unsqueeze(0)?)?; // (1, S, D)
        let text = (&text + bert)?;
        let text = (&text + self.sine_pe(x_len, text.dtype())?.broadcast_mul(&self.alpha_text)?)?;

        // 提示音频: embedding + pe
        let (xy, mask) = if y_len > 0 {
            let pids = Tensor::from_vec(prompt_codes.to_vec(), (1, y_len), &self.device)?;
            let pemb = self.audio_emb.forward(&pids)?;
            let prompt =
                (&pemb + self.sine_pe(y_len, pemb.dtype())?.broadcast_mul(&self.alpha_audio)?)?;
            let xy = Tensor::cat(&[&text, &prompt], 1)?;
            let mask = self.build_prompt_mask(x_len, y_len)?;
            (xy, Some(mask))
        } else {
            (text, None)
        };

        // 第一遍：整段 attention
        let mut caches: Vec<Option<(Tensor, Tensor)>> = (0..self.blocks.len()).map(|_| None).collect();
        let mut hidden = xy;
        for (i, block) in self.blocks.iter().enumerate() {
            let (out, k, v) = block.forward(&hidden, None, mask.as_ref())?;
            hidden = out;
            caches[i] = Some((k, v));
        }

        let mut y: Vec<u32> = prompt_codes.to_vec();
        let mut last = hidden.narrow(1, hidden.dim(1)? - 1, 1)?; // (1, 1, H)
        let mut generated = Vec::new();

        for idx in 0..opts.max_new_tokens {
            let logits = self.predict.forward(&last.squeeze(1)?)?.squeeze(0)?; // (vocab,)

            // 前 11 步屏蔽 EOS（对应 Python 的 `logits = logits[:, :-1]`）
            let candidates = if idx < 11 && self.cfg.eos + 1 == logits.dim(0)? {
                logits.narrow(0, 0, self.cfg.eos)?
            } else {
                logits.clone()
            };
            let sampled = if opts.greedy {
                candidates.argmax(0)?.to_scalar::<u32>()?
            } else {
                simple_sample(
                    &candidates,
                    true,
                    Some(opts.temperature as f64),
                    Some(opts.top_k),
                    Some(opts.top_p),
                    Some(&y),
                    opts.repetition_penalty,
                )?
            };
            let argmax = candidates.argmax(0)?.to_scalar::<u32>()?;

            let stop_by_eos = argmax as usize == self.cfg.eos || sampled as usize == self.cfg.eos;
            let stopped = stop_by_eos
                || opts
                    .early_stop_num
                    .is_some_and(|n| generated.len() + 1 > n)
                || idx + 1 >= opts.max_new_tokens;

            if !stop_by_eos {
                y.push(sampled);
                generated.push(sampled);
            }

            if stopped {
                break;
            }

            // 下一步输入：上一 token 的音频嵌入 + 位置 y_len + idx
            let next_ids = Tensor::from_vec(vec![sampled], (1, 1), &self.device)?;
            let emb = self.audio_emb.forward(&next_ids)?;
            let x = self.add_pe(&emb, &self.alpha_audio, y_len + idx)?;

            let mut next = x;
            for (i, block) in self.blocks.iter().enumerate() {
                let (kc, vc) = caches[i]
                    .as_ref()
                    .map(|(k, v)| (k, v))
                    .expect("cache 已初始化");
                let (out, k, v) = block.forward(&next, Some((kc, vc)), None)?;
                next = out;
                caches[i] = Some((k, v));
            }
            last = next;
        }

        Ok(generated)
    }

    fn build_prompt_mask(&self, x_len: usize, y_len: usize) -> Result<Tensor> {
        let total = x_len + y_len;
        let mut values = vec![0f32; total * total];
        for r in 0..total {
            for c in 0..total {
                let blocked = if r < x_len {
                    c >= x_len // 文本行看不到音频列
                } else {
                    c >= x_len && c - x_len > r - x_len // 音频行对音频列因果
                };
                if blocked {
                    values[r * total + c] = f32::NEG_INFINITY;
                }
            }
        }
        Ok(Tensor::from_vec(values, (1, 1, total, total), &self.device)?)
    }
}
