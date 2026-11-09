//! 视觉-语言融合：VisionProjection + 6 层 BiAttentionBlock + 文本 TransformerEncoderLayer。
//!
//! 移植自 `turbovla/models/components/fusion.py` 和 `transformer.py`（Grounding DINO 的
//! 双向注意力）。注意 `attention_backend="manual"`：走手写注意力（稳定 softmax + clamp），
//! 不是 sdpa。实现在 `ops.rs`：语义和 Python 一致，但不做设备同步、不多拷一份转置。

use candle_core::{DType, Module, Result, Tensor, D};
use candle_nn::{LayerNorm, LayerNormConfig, VarBuilder};
use super::linear::{linear, linear_no_bias, Linear};

use super::config::InteractionConfig;
use super::ops;

fn ln(dim: usize, vb: VarBuilder) -> Result<LayerNorm> {
    candle_nn::layer_norm(dim, LayerNormConfig { eps: 1e-5, ..Default::default() }, vb)
}

/// 768 → hidden（256）：LayerNorm + MLP + skip，再 LayerNorm。
pub struct VisionProjection {
    input_norm: LayerNorm,
    mlp0: Linear,
    mlp3: Linear,
    skip: Linear,
    output_norm: LayerNorm,
}

impl VisionProjection {
    pub fn load(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let vb_p = vb.pp("vision_projection");
        let hidden = (out_dim * 4).max(in_dim / 2);
        Ok(Self {
            input_norm: ln(in_dim, vb_p.pp("input_norm"))?,
            mlp0: linear(in_dim, hidden, vb_p.pp("mlp.0"))?,
            mlp3: linear(hidden, out_dim, vb_p.pp("mlp.3"))?,
            skip: linear_no_bias(in_dim, out_dim, vb_p.pp("skip"))?,
            output_norm: ln(out_dim, vb_p.pp("output_norm"))?,
        })
    }

    pub fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        let n = self.input_norm.forward(tokens)?;
        let m = candle_nn::Activation::Gelu.forward(&self.mlp0.forward(&n)?)?;
        let m = self.mlp3.forward(&m)?;
        let s = self.skip.forward(tokens)?;
        self.output_norm.forward(&(s + m)?)
    }
}

/// 把 packed qkv（(3d,d) / (3d,)）拆成三个 Linear。
pub(super) fn split_qkv(vb: VarBuilder, dim: usize, name: &str) -> Result<(Linear, Linear, Linear)> {
    let w = vb.get((3 * dim, dim), &format!("{name}.in_proj_weight"))?;
    let b = vb.get(3 * dim, &format!("{name}.in_proj_bias"))?;
    let part = |i: usize| -> Result<Linear> {
        let w = w.narrow(0, i * dim, dim)?.contiguous()?;
        let b = b.narrow(0, i * dim, dim)?.contiguous()?;
        Ok(Linear::new(w, Some(b)))
    };
    Ok((part(0)?, part(1)?, part(2)?))
}

/// `self_mask`：(B,S,S) u8，1 = 允许（子句级 3D mask）→ 加性 mask (B,1,S,S)。
fn self_mask_additive(self_mask: Option<&Tensor>, dtype: DType) -> Result<Option<Tensor>> {
    match self_mask {
        Some(m) => Ok(Some(ops::additive_keep(m, dtype)?.unsqueeze(1)?)),
        None => Ok(None),
    }
}

/// 文本侧的 TransformerEncoderLayer（post-norm，ReLU，packed qkv）。
pub struct TextLayer {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    linear1: Linear,
    linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    heads: usize,
    scale: f64,
}

impl TextLayer {
    pub fn load(cfg: &InteractionConfig, vb: VarBuilder) -> Result<Self> {
        let d = cfg.hidden_dim;
        let (q, k, v) = split_qkv(vb.clone(), d, "self_attn")?;
        Ok(Self {
            q,
            k,
            v,
            out: linear(d, d, vb.pp("self_attn").pp("out_proj"))?,
            linear1: linear(d, cfg.enhancer_inner_dim, vb.pp("linear1"))?,
            linear2: linear(cfg.enhancer_inner_dim, d, vb.pp("linear2"))?,
            norm1: ln(d, vb.pp("norm1"))?,
            norm2: ln(d, vb.pp("norm2"))?,
            heads: cfg.nheads / 2,
            scale: ((d / (cfg.nheads / 2)) as f64).powf(-0.5),
        })
    }

    pub fn forward(&self, src: &Tensor, self_mask: Option<&Tensor>) -> Result<Tensor> {
        let q = self.q.forward(src)?;
        let k = self.k.forward(src)?;
        let v = self.v.forward(src)?;
        let additive = self_mask_additive(self_mask, src.dtype())?;
        let attn = ops::masked_attention(&q, &k, &v, self.heads, self.scale, additive.as_ref())?;
        let attn = self.out.forward(&attn)?;
        let x = self.norm1.forward(&(src + attn)?)?;
        let mlp = candle_nn::Activation::Relu.forward(&self.linear1.forward(&x)?)?;
        let mlp = self.linear2.forward(&mlp)?;
        self.norm2.forward(&(x + mlp)?)
    }
}

/// 双向注意力（vision ↔ language），手动实现。
pub struct BiAttn {
    v_proj: Linear,
    l_proj: Linear,
    values_v: Linear,
    values_l: Linear,
    out_v: Linear,
    out_l: Linear,
    heads: usize,
    scale: f64,
}

impl BiAttn {
    pub fn load(cfg: &InteractionConfig, vb: VarBuilder) -> Result<Self> {
        let d = cfg.hidden_dim;
        let inner = cfg.enhancer_inner_dim;
        let vb_a = vb.pp("attn");
        Ok(Self {
            v_proj: linear(d, inner, vb_a.pp("v_proj"))?,
            l_proj: linear(d, inner, vb_a.pp("l_proj"))?,
            values_v: linear(d, inner, vb_a.pp("values_v_proj"))?,
            values_l: linear(d, inner, vb_a.pp("values_l_proj"))?,
            out_v: linear(inner, d, vb_a.pp("out_v_proj"))?,
            out_l: linear(inner, d, vb_a.pp("out_l_proj"))?,
            heads: (cfg.nheads / 2).max(1),
            scale: ((inner / (cfg.nheads / 2).max(1)) as f64).powf(-0.5),
        })
    }

    /// `mask_l`: (B, Sl) bool（true = 这个文本 token 是 padding）。
    pub fn forward(
        &self,
        v: &Tensor,
        l: &Tensor,
        mask_l: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let q = ops::split_heads(&self.v_proj.forward(v)?, self.heads)?;
        let k = ops::split_heads(&self.l_proj.forward(l)?, self.heads)?;
        let vv = ops::split_heads(&self.values_v.forward(v)?, self.heads)?;
        let vl = ops::split_heads(&self.values_l.forward(l)?, self.heads)?;

        let (out_v, out_l) = ops::bi_attention(&q, &k, &vv, &vl, self.scale, mask_l)?;
        Ok((self.out_v.forward(&out_v)?, self.out_l.forward(&out_l)?))
    }
}

/// BiAttentionBlock：pre-LN + 双向注意力 + LayerScale（residual_style="normalized"）。
pub struct FusionBlock {
    layer_norm_v: LayerNorm,
    layer_norm_l: LayerNorm,
    attn: BiAttn,
    gamma_v: Tensor,
    gamma_l: Tensor,
}

impl FusionBlock {
    pub fn load(cfg: &InteractionConfig, vb: VarBuilder) -> Result<Self> {
        let d = cfg.hidden_dim;
        Ok(Self {
            layer_norm_v: ln(d, vb.pp("layer_norm_v"))?,
            layer_norm_l: ln(d, vb.pp("layer_norm_l"))?,
            attn: BiAttn::load(cfg, vb.clone())?,
            gamma_v: vb.get(d, "gamma_v")?,
            gamma_l: vb.get(d, "gamma_l")?,
        })
    }

    pub fn forward(
        &self,
        v: &Tensor,
        l: &Tensor,
        mask_l: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let vn = self.layer_norm_v.forward(v)?;
        let ln = self.layer_norm_l.forward(l)?;
        let (dv, dl) = self.attn.forward(&vn, &ln, mask_l)?;
        let v_out = (vn + dv.broadcast_mul(&self.gamma_v)?)?;
        let l_out = (ln + dl.broadcast_mul(&self.gamma_l)?)?;
        Ok((v_out, l_out))
    }
}

/// 6 层融合：每层先 BiAttention，再文本 TransformerEncoderLayer。
pub struct Interaction {
    fusion: Vec<FusionBlock>,
    text: Vec<TextLayer>,
}

impl Interaction {
    pub fn load(cfg: &InteractionConfig, vb: VarBuilder) -> Result<Self> {
        let mut fusion = Vec::with_capacity(cfg.num_layers);
        let mut text = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let vb_i = vb.pp("vision_language_interaction");
            fusion.push(FusionBlock::load(cfg, vb_i.pp(format!("fusion_layers.{i}")))?);
            text.push(TextLayer::load(cfg, vb_i.pp(format!("text_layers.{i}")))?);
        }
        Ok(Self { fusion, text })
    }

    pub fn forward(
        &self,
        visual: &Tensor,
        text: &Tensor,
        mask_l: Option<&Tensor>,
        self_mask: Option<&Tensor>,
        zero_fill: bool,
    ) -> Result<(Tensor, Tensor)> {
        let mut v = visual.clone();
        let mut l = text.clone();
        for i in 0..self.fusion.len() {
            let (nv, nl) = self.forward_layer(i, &v, &l, mask_l, self_mask, zero_fill)?;
            v = nv;
            l = nl;
        }
        Ok((v, l))
    }

    /// 只跑第 `i` 层（对齐/排查用）。
    pub fn forward_layer(
        &self,
        i: usize,
        visual: &Tensor,
        text: &Tensor,
        mask_l: Option<&Tensor>,
        self_mask: Option<&Tensor>,
        zero_fill: bool,
    ) -> Result<(Tensor, Tensor)> {
        let fusion = &self.fusion[i];
        let text_layer = &self.text[i];
        let mut v = visual.clone();
        let mut l = text.clone();
        {
            if zero_fill {
                // padding_strategy="zero_fill" 时，padding 位置的文本 token 置 0
                if let Some(mask) = mask_l {
                    let keep = (mask.to_dtype(DType::F32)?.ones_like()? - mask.to_dtype(DType::F32)?)?;
                    l = l.broadcast_mul(&keep.unsqueeze(D::Minus1)?)?;
                }
            }
            let (nv, nl) = fusion.forward(&v, &l, mask_l)?;
            v = nv;
            l = text_layer.forward(&nl, self_mask)?;
            if zero_fill {
                if let Some(mask) = mask_l {
                    let keep = (mask.to_dtype(DType::F32)?.ones_like()? - mask.to_dtype(DType::F32)?)?;
                    l = l.broadcast_mul(&keep.unsqueeze(D::Minus1)?)?;
                }
            }
        }
        Ok((v, l))
    }
}
