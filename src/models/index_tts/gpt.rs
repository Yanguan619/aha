//! IndexTTS 2.5 的 GPT 主干（Tortoise / UnifiedVoice 风格）。
//!
//! 和普通 GPT-2 的区别：
//! * 用**分开的** text / mel 词表和位置 embedding，而不是一个 wte/wpe；
//! * 权重是 PyTorch GPT-2 `Conv1D` 的排布 `(in, out)`（GGUF 维度翻转后就是
//!   `(in, out)`），而 candle 的 `Linear` 要 `(out, in)`，所以这四个投影统一转置；
//! * 外部还有 `final_norm` 和 `spk_emb_proj` / `lang_embedding`，
//!   conformer_perceiver 条件模块（下一阶段补）。

use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{Embedding, Init, LayerNorm, Linear, Module, VarBuilder};

use crate::models::common::modules::{eager_attention_forward, get_layer_norm};
use crate::models::index_tts::config::GptConfig;
use crate::utils::tensor_utils::prepare_causal_attention_mask;

/// GPT-2 `Conv1D` 风格的线性层：权重存的是 `(in, out)`，这里转置成 `(out, in)`
fn conv1d_linear(vb: &VarBuilder, in_dim: usize, out_dim: usize) -> Result<Linear> {
    let weight = vb
        .get_with_hints((in_dim, out_dim), "weight", Init::Const(0.0))?
        .t()?
        .contiguous()?;
    let bias = vb.get_with_hints(out_dim, "bias", Init::Const(0.0))?;
    Ok(Linear::new(weight, Some(bias)))
}

/// 普通 `nn.Linear`：权重已经是 `(out, in)`，直接按形状取
fn linear_auto(vb: &VarBuilder, name: &str) -> Result<Linear> {
    let weight = vb.get_unchecked(&format!("{name}.weight"))?;
    let (out_dim, _) = weight.dims2()?;
    let bias = vb.get_with_hints(out_dim, &format!("{name}.bias"), Init::Const(0.0))?;
    Ok(Linear::new(weight, Some(bias)))
}

/// 按权重实际形状建 embedding（IndexTTS 的词表比 config 里写的大一点点）
fn embedding_auto(vb: &VarBuilder, name: &str) -> Result<Embedding> {
    let weight = vb.get_unchecked(&format!("{name}.weight"))?;
    let (_, dim) = weight.dims2()?;
    Ok(Embedding::new(weight, dim))
}

struct GptAttention {
    n_head: usize,
    head_dim: usize,
    c_attn: Linear,
    c_proj: Linear,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl GptAttention {
    fn new(vb: &VarBuilder, dim: usize, n_head: usize) -> Result<Self> {
        Ok(Self {
            n_head,
            head_dim: dim / n_head,
            c_attn: conv1d_linear(&vb.pp("c_attn"), dim, 3 * dim)?,
            c_proj: conv1d_linear(&vb.pp("c_proj"), dim, dim)?,
            kv_cache: None,
        })
    }

    fn forward(&mut self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, seq_len, _) = xs.dims3()?;
        let qkv = self.c_attn.forward(xs)?;
        let parts = qkv.chunk(3, 2)?;
        let split = |t: &Tensor| -> Result<Tensor> {
            Ok(t.reshape((b, seq_len, self.n_head, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?)
        };
        let query = split(&parts[0])?;
        let key = split(&parts[1])?;
        let value = split(&parts[2])?;

        let (key, value) = match &self.kv_cache {
            None => (key, value),
            Some((k, v)) => (Tensor::cat(&[k, &key], 2)?, Tensor::cat(&[v, &value], 2)?),
        };
        self.kv_cache = Some((key.clone(), value.clone()));

        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let out = eager_attention_forward(&query, &key, &value, None, attention_mask, scale)?;
        let out = out.reshape((b, seq_len, self.n_head * self.head_dim))?;
        Ok(out.apply(&self.c_proj)?)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }

    /// beam search 重排：按 beam 下标重放 KV cache（k/v 第一维是 batch）
    fn reorder_kv_cache(&mut self, idx: &Tensor) -> Result<()> {
        if let Some((k, v)) = self.kv_cache.take() {
            let k = k.index_select(idx, 0)?;
            let v = v.index_select(idx, 0)?;
            self.kv_cache = Some((k, v));
        }
        Ok(())
    }
}

struct GptBlock {
    ln_1: LayerNorm,
    attn: GptAttention,
    ln_2: LayerNorm,
    c_fc: Linear,
    c_proj: Linear,
}

impl GptBlock {
    fn new(vb: &VarBuilder, dim: usize, n_head: usize, inner: usize) -> Result<Self> {
        Ok(Self {
            ln_1: get_layer_norm(vb.pp("ln_1"), 1e-5, dim, true)?,
            attn: GptAttention::new(&vb.pp("attn"), dim, n_head)?,
            ln_2: get_layer_norm(vb.pp("ln_2"), 1e-5, dim, true)?,
            c_fc: conv1d_linear(&vb.pp("mlp").pp("c_fc"), dim, inner)?,
            c_proj: conv1d_linear(&vb.pp("mlp").pp("c_proj"), inner, dim)?,
        })
    }

    fn forward(&mut self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = xs.clone();
        let h = self.ln_1.forward(xs)?;
        let h = self.attn.forward(&h, attention_mask)?;
        let xs = (h + residual)?;

        let residual = xs.clone();
        let h = self.ln_2.forward(&xs)?;
        let h = self.c_fc.forward(&h)?;
        let h = h.gelu()?;
        let h = self.c_proj.forward(&h)?;
        Ok((h + residual)?)
    }
}

pub struct IndexTtsGpt {
    pub config: GptConfig,
    text_embedding: Embedding,
    mel_embedding: Embedding,
    text_pos_embedding: Embedding,
    mel_pos_embedding: Embedding,
    pub lang_embedding: Embedding,
    pub spk_emb_proj: Linear,
    h: Vec<GptBlock>,
    ln_f: LayerNorm,
    final_norm: LayerNorm,
    text_head: Linear,
    mel_head: Linear,
}

impl IndexTtsGpt {
    pub fn new(vb: &VarBuilder, config: &GptConfig) -> Result<Self> {
        let dim = config.model_dim;
        let inner = 4 * dim;
        let vb_gpt = vb.pp("gpt");

        let text_embedding = embedding_auto(vb, "text_embedding")?;
        let mel_embedding = embedding_auto(vb, "mel_embedding")?;
        let text_pos_embedding = embedding_auto(vb, "text_pos_embedding.emb")?;
        let mel_pos_embedding = embedding_auto(vb, "mel_pos_embedding.emb")?;
        let lang_embedding = embedding_auto(vb, "lang_embedding")?;
        let spk_emb_proj = linear_auto(vb, "spk_emb_proj")?;

        let vb_h = vb_gpt.pp("h");
        let mut h = Vec::with_capacity(config.layers);
        for i in 0..config.layers {
            h.push(GptBlock::new(&vb_h.pp(i), dim, config.heads, inner)?);
        }

        let ln_f = get_layer_norm(vb_gpt.pp("ln_f"), 1e-5, dim, true)?;
        let final_norm = get_layer_norm(vb.pp("final_norm"), 1e-5, dim, true)?;
        let text_head = linear_auto(vb, "text_head")?;
        let mel_head = linear_auto(vb, "mel_head")?;

        Ok(Self {
            config: config.clone(),
            text_embedding,
            mel_embedding,
            text_pos_embedding,
            mel_pos_embedding,
            lang_embedding,
            spk_emb_proj,
            h,
            ln_f,
            final_norm,
            text_head,
            mel_head,
        })
    }

    /// 文本 token -> embedding（含位置编码）
    pub fn embed_text(&self, ids: &Tensor) -> Result<Tensor> {
        let (_, seq_len) = ids.dims2()?;
        let pos = Tensor::arange(0u32, seq_len as u32, ids.device())?.unsqueeze(0)?;
        let xs = self.text_embedding.forward(ids)?;
        let ps = self.text_pos_embedding.forward(&pos)?;
        Ok(xs.broadcast_add(&ps)?)
    }

    /// 文本 token -> embedding（再叠加语言 embedding，2.5 的 `lang_embedding`）
    pub fn embed_text_with_lang(&self, ids: &Tensor, lang_id: u32) -> Result<Tensor> {
        let xs = self.embed_text(ids)?;
        let lang = Tensor::from_vec(vec![lang_id], (1,), ids.device())?;
        let l = self.lang_embedding.forward(&lang)?; // (1, dim)
        Ok(xs.broadcast_add(&l)?)
    }

    /// mel code -> embedding（含位置编码，位置从 `offset` 开始，配合 KV cache）
    pub fn embed_mel(&self, ids: &Tensor, offset: usize) -> Result<Tensor> {
        let (_, seq_len) = ids.dims2()?;
        let pos =
            Tensor::arange(offset as u32, (offset + seq_len) as u32, ids.device())?.unsqueeze(0)?;
        let xs = self.mel_embedding.forward(ids)?;
        let ps = self.mel_pos_embedding.forward(&pos)?;
        Ok(xs.broadcast_add(&ps)?)
    }

    pub fn forward(&mut self, inputs_embeds: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b_size, seq_len, _) = inputs_embeds.dims3()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(prepare_causal_attention_mask(
                b_size,
                seq_len,
                0,
                inputs_embeds.device(),
            )?)
        };
        let mut xs = inputs_embeds.clone();
        for block in self.h.iter_mut() {
            xs = block.forward(&xs, attention_mask.as_ref())?;
        }
        let _ = seqlen_offset;
        Ok(self.ln_f.forward(&xs)?)
    }

    /// mel 头 logits：(b, seq, number_mel_codes)
    pub fn mel_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        let xs = self.final_norm.forward(hidden)?;
        Ok(self.mel_head.forward(&xs)?)
    }

    /// text 头 logits：(b, seq, number_text_tokens)
    pub fn text_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        let xs = self.final_norm.forward(hidden)?;
        Ok(self.text_head.forward(&xs)?)
    }

    pub fn clear_kv_cache(&mut self) {
        for block in self.h.iter_mut() {
            block.attn.clear_kv_cache();
        }
    }

    /// beam search：按 beam 下标重排所有层的 KV cache
    pub fn reorder_kv_cache(&mut self, idx: &Tensor) -> Result<()> {
        for block in self.h.iter_mut() {
            block.attn.reorder_kv_cache(idx)?;
        }
        Ok(())
    }
}
