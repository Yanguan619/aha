//! chinese-roberta-wwm-ext-large 的 encoder 部分（BERT 结构）。
//!
//! GPT-SoVITS 只用到 **第 21 层**（24 层里的第 22 层，对应 HF 的
//! `output_hidden_states=True` 后 `hidden_states[-3]`），并丢掉首尾的 `[CLS]`/`[SEP]`，
//! 再按 `word2ph` 把「字级」特征复制到「phone 级」。

use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Activation, Embedding, LayerNorm, Linear, VarBuilder, embedding, layer_norm};
use serde::Deserialize;

use super::wordpiece::WordPieceTokenizer;

/// `hidden_states[-3]`：embeddings 是第 0 项，所以对应第 21 层的输出。
pub const FEATURE_LAYER: usize = 21;

#[derive(Debug, Deserialize)]
pub struct BertConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub layer_norm_eps: f64,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "one")]
    pub type_vocab_size: usize,
}

fn one() -> usize {
    1
}

impl BertConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        serde_json::from_reader(std::io::BufReader::new(
            std::fs::File::open(path).with_context(|| format!("打开 {}", path.display()))?,
        ))
        .with_context(|| format!("解析 {}", path.display()))
    }
}

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(cfg: &BertConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            q: candle_nn::linear(cfg.hidden_size, cfg.hidden_size, vb.pp("self.query"))?,
            k: candle_nn::linear(cfg.hidden_size, cfg.hidden_size, vb.pp("self.key"))?,
            v: candle_nn::linear(cfg.hidden_size, cfg.hidden_size, vb.pp("self.value"))?,
            o: candle_nn::linear(cfg.hidden_size, cfg.hidden_size, vb.pp("output.dense"))?,
            heads: cfg.num_attention_heads,
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let shape = (b, s, self.heads, self.head_dim);
        // CUDA 的 matmul 只接受连续张量，transpose 后必须 contiguous
        let q = self.q.forward(x)?.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        let k = self.k.forward(x)?.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        let v = self.v.forward(x)?.reshape(shape)?.transpose(1, 2)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = probs.matmul(&v)?;
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, s, self.heads * self.head_dim))?;
        Ok(self.o.forward(&out)?)
    }
}

struct EncoderLayer {
    attn: Attention,
    ln1: LayerNorm,
    ff1: Linear,
    ff2: Linear,
    ln2: LayerNorm,
}

impl EncoderLayer {
    fn new(cfg: &BertConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn: Attention::new(cfg, vb.pp("attention"))?,
            ln1: layer_norm(
                cfg.hidden_size,
                cfg.layer_norm_eps,
                vb.pp("attention.output.LayerNorm"),
            )?,
            ff1: candle_nn::linear(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("intermediate.dense"),
            )?,
            ff2: candle_nn::linear(
                cfg.intermediate_size,
                cfg.hidden_size,
                vb.pp("output.dense"),
            )?,
            ln2: layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("output.LayerNorm"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let attn = self.attn.forward(x)?;
        let x = self.ln1.forward(&(x + attn)?)?;
        let h = Activation::Gelu.forward(&self.ff1.forward(&x)?)?;
        let h = self.ff2.forward(&h)?;
        Ok(self.ln2.forward(&(&x + h)?)?)
    }
}

pub struct RobertaEncoder {
    wte: Embedding,
    wpe: Embedding,
    tte: Embedding,
    ln: LayerNorm,
    layers: Vec<EncoderLayer>,
    device: Device,
    dtype: DType,
    feature_layer: usize,
}

impl RobertaEncoder {
    pub fn load(dir: impl AsRef<Path>, device: &Device, dtype: DType) -> Result<Self> {
        let dir = dir.as_ref();
        let cfg = BertConfig::load(dir.join("bert_config.json"))?;
        let tensors = candle_core::safetensors::load(dir.join("bert.safetensors"), device)?;
        let vb = VarBuilder::from_tensors(tensors, dtype, device);
        let base = vb.pp("bert");
        let emb = base.pp("embeddings");

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(EncoderLayer::new(&cfg, base.pp("encoder").pp("layer").pp(i))?);
        }

        Ok(Self {
            wte: embedding(cfg.vocab_size, cfg.hidden_size, emb.pp("word_embeddings"))?,
            wpe: embedding(
                cfg.max_position_embeddings,
                cfg.hidden_size,
                emb.pp("position_embeddings"),
            )?,
            tte: embedding(
                cfg.type_vocab_size,
                cfg.hidden_size,
                emb.pp("token_type_embeddings"),
            )?,
            ln: layer_norm(cfg.hidden_size, cfg.layer_norm_eps, emb.pp("LayerNorm"))?,
            layers,
            device: device.clone(),
            dtype,
            feature_layer: FEATURE_LAYER,
        })
    }

    pub fn with_feature_layer(mut self, layer: usize) -> Self {
        self.feature_layer = layer;
        self
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// `ids` → (1, seq, hidden)，返回第 `feature_layer` 层的输出。
    pub fn forward(&self, ids: &[u32]) -> Result<Tensor> {
        let seq = ids.len();
        let ids = Tensor::from_vec(ids.to_vec(), (1, seq), &self.device)?;
        let positions =
            Tensor::from_vec((0..seq as u32).collect::<Vec<_>>(), (1, seq), &self.device)?;
        let token_types = Tensor::zeros((1, seq), DType::U32, &self.device)?;

        let mut x = ((self.wte.forward(&ids)? + self.wpe.forward(&positions)?)?
            + self.tte.forward(&token_types)?)?;
        x = self.ln.forward(&x)?;

        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x)?;
            if i >= self.feature_layer {
                break;
            }
        }
        Ok(x)
    }

    /// GPT-SoVITS 的 `get_bert_feature`：字级特征 → phone 级 `(hidden, n_phones)`。
    pub fn phone_level_feature(
        &self,
        text: &str,
        tokenizer: &WordPieceTokenizer,
        word2ph: &[usize],
    ) -> Result<Tensor> {
        let ids = tokenizer.encode(text);
        let hidden = self.forward(&ids)?; // (1, 1 + chars + 1, H)
        let n = hidden.dim(1)?;
        if n < 2 || word2ph.len() != n - 2 {
            anyhow::bail!(
                "bert 特征长度 {} 与 word2ph 长度 {} 不匹配（文本 {text:?}）",
                n.saturating_sub(2),
                word2ph.len()
            );
        }
        // 去掉 [CLS] / [SEP]
        let chars = hidden.narrow(1, 1, n - 2)?.squeeze(0)?; // (chars, H)
        let total: usize = word2ph.iter().sum();
        let mut rows = Vec::with_capacity(total);
        for (i, &repeat) in word2ph.iter().enumerate() {
            let row = chars.narrow(0, i, 1)?;
            for _ in 0..repeat {
                rows.push(row.clone());
            }
        }
        let out = Tensor::cat(&rows, 0)?; // (phones, H)
        Ok(out.t()?.contiguous()?)
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}
