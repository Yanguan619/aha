//! 动作头：StateProjection + 3 层 TransformerDecoder（12 个 query → 12×7，tanh）。
//!
//! 移植自 `turbovla/models/action_head.py`；decoder 层是 PyTorch
//! `TransformerDecoderLayer(batch_first=True, norm_first=True)`。

use candle_core::{Module, Result, Tensor};
use candle_nn::{LayerNorm, LayerNormConfig, Linear, VarBuilder};

use super::config::ActionConfig;
use super::fusion::split_qkv;

fn ln(dim: usize, vb: VarBuilder) -> Result<LayerNorm> {
    candle_nn::layer_norm(dim, LayerNormConfig { eps: 1e-5, ..Default::default() }, vb)
}

/// 8 维 state → 2 个 token（256）。
pub struct StateProjection {
    input_norm: LayerNorm,
    l1: Linear,
    l4: Linear,
    position: Tensor,
    output_norm: LayerNorm,
    tokens: usize,
}

impl StateProjection {
    pub fn load(cfg: &ActionConfig, hidden: usize, vb: VarBuilder) -> Result<Self> {
        let vb_p = vb.pp("state_projection");
        let tokens = cfg.num_state_tokens;
        Ok(Self {
            input_norm: ln(cfg.state_dim, vb_p.pp("net.0"))?,
            l1: candle_nn::linear(cfg.state_dim, cfg.state_hidden_dim, vb_p.pp("net.1"))?,
            l4: candle_nn::linear(cfg.state_hidden_dim, tokens * hidden, vb_p.pp("net.4"))?,
            position: vb_p.get((1, tokens, hidden), "position")?,
            output_norm: ln(hidden, vb_p.pp("output_norm"))?,
            tokens,
        })
    }

    pub fn forward(&self, state: &Tensor, hidden: usize) -> Result<Tensor> {
        let (b, _) = state.dims2()?;
        let x = self.input_norm.forward(state)?;
        let x = candle_nn::Activation::Gelu.forward(&self.l1.forward(&x)?)?;
        let x = self.l4.forward(&x)?;
        let x = x.reshape((b, self.tokens, hidden))?;
        self.output_norm.forward(&x.broadcast_add(&self.position)?)
    }
}

struct DecoderLayer {
    self_q: Linear,
    self_k: Linear,
    self_v: Linear,
    self_out: Linear,
    cross_q: Linear,
    cross_k: Linear,
    cross_v: Linear,
    cross_out: Linear,
    linear1: Linear,
    linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
    heads: usize,
    scale: f64,
}

impl DecoderLayer {
    fn load(cfg: &ActionConfig, hidden: usize, nheads: usize, ff: usize, vb: VarBuilder) -> Result<Self> {
        let _ = cfg;
        let (self_q, self_k, self_v) = split_qkv(vb.clone(), hidden, "self_attn")?;
        let (cross_q, cross_k, cross_v) = split_qkv(vb.clone(), hidden, "multihead_attn")?;
        Ok(Self {
            self_q,
            self_k,
            self_v,
            self_out: candle_nn::linear(hidden, hidden, vb.pp("self_attn").pp("out_proj"))?,
            cross_q,
            cross_k,
            cross_v,
            cross_out: candle_nn::linear(hidden, hidden, vb.pp("multihead_attn").pp("out_proj"))?,
            linear1: candle_nn::linear(hidden, ff, vb.pp("linear1"))?,
            linear2: candle_nn::linear(ff, hidden, vb.pp("linear2"))?,
            norm1: ln(hidden, vb.pp("norm1"))?,
            norm2: ln(hidden, vb.pp("norm2"))?,
            norm3: ln(hidden, vb.pp("norm3"))?,
            heads: nheads,
            scale: ((hidden / nheads) as f64).powf(-0.5),
        })
    }

    fn attn(&self, q: &Tensor, k: &Tensor, v: &Tensor, heads: usize, scale: f64) -> Result<Tensor> {
        let (b, sq, _) = q.dims3()?;
        let sk = k.dim(1)?;
        let hd = q.dim(2)? / heads;
        let q = q.reshape((b, sq, heads, hd))?.transpose(1, 2)?.contiguous()?;
        let k = k.reshape((b, sk, heads, hd))?.transpose(1, 2)?.contiguous()?;
        let v = v.reshape((b, sk, heads, hd))?.transpose(1, 2)?.contiguous()?;
        let a = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let a = candle_nn::ops::softmax_last_dim(&a)?;
        a.matmul(&v)?.transpose(1, 2)?.reshape((b, sq, heads * hd))
    }

    /// norm_first=True 的 decoder 层。
    fn forward(&self, tgt: &Tensor, memory: &Tensor) -> Result<Tensor> {
        let x = self.norm1.forward(tgt)?;
        let a = self.attn(&self.self_q.forward(&x)?, &self.self_k.forward(&x)?, &self.self_v.forward(&x)?, self.heads, self.scale)?;
        let tgt = (tgt + self.self_out.forward(&a)?)?;

        let x = self.norm2.forward(&tgt)?;
        let a = self.attn(
            &self.cross_q.forward(&x)?,
            &self.cross_k.forward(memory)?,
            &self.cross_v.forward(memory)?,
            self.heads,
            self.scale,
        )?;
        let tgt = (tgt + self.cross_out.forward(&a)?)?;

        let x = self.norm3.forward(&tgt)?;
        let x = candle_nn::Activation::Relu.forward(&self.linear1.forward(&x)?)?;
        let x = self.linear2.forward(&x)?;
        Ok((tgt + x)?)
    }
}

/// 动作 chunk 解码器。
pub struct ActionHead {
    state: StateProjection,
    queries: Tensor,
    layers: Vec<DecoderLayer>,
    proj: [Linear; 3],
    hidden: usize,
    horizon: usize,
}

impl ActionHead {
    pub fn load(
        cfg: &ActionConfig,
        hidden: usize,
        nheads: usize,
        ff: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let state = StateProjection::load(cfg, hidden, vb.clone())?;
        let vb_d = vb.pp("decoder");
        let queries = vb_d.get((cfg.horizon, hidden), "action_queries.weight")?;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(DecoderLayer::load(cfg, hidden, nheads, ff, vb_d.pp(format!("decoder.layers.{i}")))?);
        }
        let mid = cfg.mlp_hidden_dim;
        let vb_p = vb_d.pp("action_projection");
        Ok(Self {
            state,
            queries,
            layers,
            proj: [
                candle_nn::linear(hidden, mid, vb_p.pp("layers.0"))?,
                candle_nn::linear(mid, mid, vb_p.pp("layers.1"))?,
                candle_nn::linear(mid, cfg.action_dim, vb_p.pp("layers.2"))?,
            ],
            hidden,
            horizon: cfg.horizon,
        })
    }

    pub fn forward(&self, condition: &Tensor, state: &Tensor) -> Result<Tensor> {
        let (b, _, _) = condition.dims3()?;
        let st = self.state.forward(state, self.hidden)?;
        let memory = Tensor::cat(&[condition, &st], 1)?;
        let mut x = self.queries.unsqueeze(0)?.broadcast_as((b, self.horizon, self.hidden))?;
        for layer in &self.layers {
            x = layer.forward(&x, &memory)?;
        }
        let x = candle_nn::Activation::Relu.forward(&self.proj[0].forward(&x)?)?;
        let x = candle_nn::Activation::Relu.forward(&self.proj[1].forward(&x)?)?;
        let x = self.proj[2].forward(&x)?;
        x.tanh()
    }
}
