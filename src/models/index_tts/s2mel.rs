//! IndexTTS 2.5 的 s2mel（semantic → mel）模块。
//!
//! 结构（对照上游 `indextts/s2mel/modules/` 下 `commons.py::MyModel`、
//! `length_regulator.py`、`diffusion_transformer.py`、`gpt_fast/model.py`、`wavenet.py`、
//! `flow_matching.py`）：
//!
//! ```text
//! GPT hidden (b,T,1280) ── gpt_layer(1280→256→128→1024) ─┐
//!                                                        ├─ 加到语义特征上
//! semantic_codec 解码特征 (b,T,1024) ────────────────────┘
//!   └─ length_regulator: content_in_proj(1024→512) → interpolate → 4×(conv3/GN/Mish) → conv1 → (b,T',512)
//!   └─ cfm.inference: Euler + CFG，estimator = U-DiT(13 层, 512 维, uvit skip, AdaLN, wavenet 尾)
//!        └─ (b,80,T') mel → bigvgan → wav
//! ```

use anyhow::Result;
use candle_core::{D, Device, Tensor};
use candle_nn::{Conv1d, LayerNorm, Linear, Module, VarBuilder, group_norm, linear};

use crate::models::common::modules::{WNConv1d, get_layer_norm_without_weight, mish};

// ---------------------------------------------------------------- 基础组件

/// 正弦时间步嵌入（`scale=1000`、`max_period=10000`、`cos` 在前）
struct TimestepEmbedder {
    mlp_0: Linear,
    mlp_2: Linear,
    freqs: Tensor,
    scale: f64,
}

impl TimestepEmbedder {
    fn new(vb: &VarBuilder, hidden: usize) -> Result<Self> {
        let freq_dim = vb.get_unchecked("mlp.0.weight")?.dim(1)?;
        let half = freq_dim / 2;
        let freqs = vb
            .get_unchecked("freqs")?
            .to_dtype(candle_core::DType::F32)?;
        Ok(Self {
            mlp_0: linear(freq_dim, hidden, vb.pp("mlp").pp(0))?,
            mlp_2: linear(hidden, hidden, vb.pp("mlp").pp(2))?,
            freqs: freqs.reshape((half,))?,
            scale: 1000.0,
        })
    }

    fn forward(&self, t: &Tensor) -> Result<Tensor> {
        // t: (b,) -> (b, freq_dim)
        let t = t.to_dtype(candle_core::DType::F32)?.unsqueeze(1)?; // (b,1)
        let args = (t * self.scale)?.broadcast_mul(&self.freqs.unsqueeze(0)?)?; // (b,half)
        let emb = Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)?;
        let emb = emb.to_dtype(self.mlp_0.weight().dtype())?;
        let xs = self.mlp_0.forward(&emb)?.silu()?;
        Ok(self.mlp_2.forward(&xs)?)
    }
}

/// 不带仿射的 RMSNorm（`nn.Parameter(ones)` 那只 weight 单独取）
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(vb: &VarBuilder, dim: usize, eps: f64) -> Result<Self> {
        Ok(Self {
            weight: vb.get(dim, "weight")?,
            eps,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = xs.to_dtype(candle_core::DType::F32)?;
        let denom = (x.sqr()?.mean_keepdim(D::Minus1)? + self.eps)?.sqrt()?;
        let x = x.broadcast_div(&denom)?;
        let x = x.to_dtype(self.weight.dtype())?;
        Ok(x.broadcast_mul(&self.weight)?)
    }
}

enum AdaptedNorm {
    Rms(RmsNorm),
}

/// AdaptiveLayerNorm：`project_layer(c)` 出 weight/bias，做 `weight * norm(x) + bias`
struct AdaptiveLayerNorm {
    project_layer: Linear,
    norm: AdaptedNorm,
    dim: usize,
}

impl AdaptiveLayerNorm {
    fn new_rms(vb: &VarBuilder, dim: usize, eps: f64) -> Result<Self> {
        Ok(Self {
            project_layer: linear(dim, 2 * dim, vb.pp("project_layer"))?,
            norm: AdaptedNorm::Rms(RmsNorm::new(&vb.pp("norm"), dim, eps)?),
            dim,
        })
    }

    fn forward(&self, xs: &Tensor, cond: Option<&Tensor>) -> Result<Tensor> {
        let normed = match &self.norm {
            AdaptedNorm::Rms(n) => n.forward(xs)?,
        };
        let Some(c) = cond else {
            return Ok(normed);
        };
        let projected = self.project_layer.forward(c)?;
        let weight = projected.narrow(D::Minus1, 0, self.dim)?;
        let bias = projected.narrow(D::Minus1, self.dim, self.dim)?;
        Ok(normed.broadcast_mul(&weight)?.broadcast_add(&bias)?)
    }
}

/// linear + weight_norm（GGUF 里是 `weight_g` / `weight_v` 两个张量）
fn linear_weight_norm(vb: &VarBuilder, in_dim: usize, out_dim: usize) -> Result<Linear> {
    let g = vb.get((out_dim, 1), "weight_g")?;
    let v = vb.get((out_dim, in_dim), "weight_v")?;
    let norm = v.sqr()?.sum_keepdim(1)?.sqrt()?;
    let weight = v.broadcast_div(&norm)?.broadcast_mul(&g)?.contiguous()?;
    let bias = vb.get(out_dim, "bias").ok();
    Ok(Linear::new(weight, bias))
}

/// 上游的 `apply_rotary_emb`（交错式：把最后一维拆成 (d/2, 2)）
fn rope(xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b, t, h, d) = xs.dims4()?;
    let half = d / 2;
    let x = xs.reshape((b, t, h, half, 2))?;
    let x0 = x.narrow(D::Minus1, 0, 1)?.squeeze(D::Minus1)?;
    let x1 = x.narrow(D::Minus1, 1, 1)?.squeeze(D::Minus1)?;
    let cos = cos.reshape((1, t, 1, half))?;
    let sin = sin.reshape((1, t, 1, half))?;
    let o0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
    let o1 = (x1.broadcast_mul(&cos)? + x0.broadcast_mul(&sin)?)?;
    Ok(Tensor::stack(&[o0, o1], D::Minus1)?
        .reshape((b, t, h, d))?
        .contiguous()?)
}

fn precompute_rope(
    max_len: usize,
    head_dim: usize,
    device: &Device,
    dtype: candle_core::DType,
) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let mut cos = vec![0f32; max_len * half];
    let mut sin = vec![0f32; max_len * half];
    for pos in 0..max_len {
        for i in 0..half {
            let angle = pos as f64 / 10000f64.powf((2 * i) as f64 / head_dim as f64);
            cos[pos * half + i] = angle.cos() as f32;
            sin[pos * half + i] = angle.sin() as f32;
        }
    }
    Ok((
        Tensor::from_vec(cos, (max_len, half), device)?.to_dtype(dtype)?,
        Tensor::from_vec(sin, (max_len, half), device)?.to_dtype(dtype)?,
    ))
}

/// DiT 里的 attention：wqkv 一次出 qkv，RoPE 后做全（非因果）注意力
struct DitAttention {
    n_head: usize,
    n_local_heads: usize,
    head_dim: usize,
    wqkv: Linear,
    wo: Linear,
}

impl DitAttention {
    fn new(vb: &VarBuilder, dim: usize, n_head: usize, head_dim: usize) -> Result<Self> {
        Ok(Self {
            n_head,
            n_local_heads: n_head,
            head_dim,
            wqkv: linear_no_bias(dim, (n_head + 2 * n_head) * head_dim, vb.pp("wqkv"))?,
            wo: linear_no_bias(n_head * head_dim, dim, vb.pp("wo"))?,
        })
    }

    fn forward(&self, xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (b, t, _) = xs.dims3()?;
        let (h, hd) = (self.n_head, self.head_dim);
        let qkv = self.wqkv.forward(xs)?;
        let parts = qkv.chunk(3, D::Minus1)?;
        let shape = (b, t, h, hd);
        let q = parts[0].reshape(shape)?.contiguous()?;
        let k = parts[1].reshape(shape)?.contiguous()?;
        let v = parts[2].reshape(shape)?.transpose(1, 2)?.contiguous()?;

        let q = rope(&q, cos, sin)?.transpose(1, 2)?.contiguous()?;
        let k = rope(&k, cos, sin)?
            .repeat((1, 1, self.n_head / self.n_local_heads, 1))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v.repeat((1, self.n_head / self.n_local_heads, 1, 1))?;

        let scale = 1f64 / f64::sqrt(hd as f64);
        let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = attn.matmul(&v)?;
        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, t, h * hd))?;
        Ok(self.wo.forward(&out)?)
    }
}

fn linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    Ok(Linear::new(vb.get((out_dim, in_dim), "weight")?, None))
}

struct DitFeedForward {
    w1: Linear,
    w3: Linear,
    w2: Linear,
}

impl DitFeedForward {
    fn new(vb: &VarBuilder, dim: usize, intermediate: usize) -> Result<Self> {
        Ok(Self {
            w1: linear_no_bias(dim, intermediate, vb.pp("w1"))?,
            w3: linear_no_bias(dim, intermediate, vb.pp("w3"))?,
            w2: linear_no_bias(intermediate, dim, vb.pp("w2"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let a = self.w1.forward(xs)?.silu()?;
        let b = self.w3.forward(xs)?;
        Ok(self.w2.forward(&(a * b)?)?)
    }
}

struct DitBlock {
    attention_norm: AdaptiveLayerNorm,
    attention: DitAttention,
    ffn_norm: AdaptiveLayerNorm,
    feed_forward: DitFeedForward,
    skip_in_linear: Option<Linear>,
}

impl DitBlock {
    fn new(
        vb: &VarBuilder,
        dim: usize,
        n_head: usize,
        head_dim: usize,
        intermediate: usize,
    ) -> Result<Self> {
        let skip_in_linear = match vb.contains_tensor("skip_in_linear.weight") {
            true => Some(linear(dim * 2, dim, vb.pp("skip_in_linear"))?),
            false => None,
        };
        Ok(Self {
            attention_norm: AdaptiveLayerNorm::new_rms(&vb.pp("attention_norm"), dim, 1e-5)?,
            attention: DitAttention::new(&vb.pp("attention"), dim, n_head, head_dim)?,
            ffn_norm: AdaptiveLayerNorm::new_rms(&vb.pp("ffn_norm"), dim, 1e-5)?,
            feed_forward: DitFeedForward::new(&vb.pp("feed_forward"), dim, intermediate)?,
            skip_in_linear,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        c: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        skip_in: Option<&Tensor>,
    ) -> Result<Tensor> {
        let xs = match (skip_in, &self.skip_in_linear) {
            (Some(skip), Some(lin)) => lin.forward(&Tensor::cat(&[xs, skip], D::Minus1)?)?,
            _ => xs.clone(),
        };
        let h = self.attention_norm.forward(&xs, Some(c))?;
        let h = (self.attention.forward(&h, cos, sin)? + &xs)?;
        let f = self.ffn_norm.forward(&h, Some(c))?;
        Ok((self.feed_forward.forward(&f)? + &h)?)
    }
}

/// wavenet 的最终层（AdaIN + weight_norm Linear）
struct DitFinalLayer {
    linear: Linear,
    adaln: Linear,
    norm_final: LayerNorm,
    hidden: usize,
}

impl DitFinalLayer {
    fn new(vb: &VarBuilder, hidden: usize) -> Result<Self> {
        Ok(Self {
            linear: linear_weight_norm(&vb.pp("linear"), hidden, hidden)?,
            adaln: linear(hidden, 2 * hidden, vb.pp("adaLN_modulation").pp(1))?,
            // elementwise_affine=False, eps=1e-6
            norm_final: get_layer_norm_without_weight(vb.pp("norm_final"), 1e-6, hidden)?,
            hidden,
        })
    }

    fn forward(&self, xs: &Tensor, c: &Tensor) -> Result<Tensor> {
        let modulation = self.adaln.forward(&c.silu()?)?;
        let shift = modulation.narrow(D::Minus1, 0, self.hidden)?;
        let scale = modulation.narrow(D::Minus1, self.hidden, self.hidden)?;
        let x = self.norm_final.forward(xs)?;
        let x = x.broadcast_mul(&(scale.unsqueeze(1)? + 1.0)?)?;
        let x = x.broadcast_add(&shift.unsqueeze(1)?)?;
        Ok(self.linear.forward(&x)?)
    }
}

/// wavenet（VITS 的 WN）
struct Wavenet {
    in_layers: Vec<WNConv1d>,
    res_skip_layers: Vec<WNConv1d>,
    cond_layer: WNConv1d,
    hidden: usize,
    n_layers: usize,
}

impl Wavenet {
    fn new(
        vb: &VarBuilder,
        hidden: usize,
        kernel: usize,
        n_layers: usize,
        gin: usize,
    ) -> Result<Self> {
        let mut in_layers = Vec::with_capacity(n_layers);
        let mut res_skip_layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let dilation = 1usize << i; // dilation_rate = 1
            let padding = (kernel * dilation - dilation) / 2;
            in_layers.push(WNConv1d::new(
                vb.pp("in_layers").pp(i).pp("conv").pp("conv"),
                hidden,
                2 * hidden,
                kernel,
                dilation,
                padding,
                1,
                1,
                true,
                None,
                None,
            )?);
            let out_c = if i < n_layers - 1 { 2 * hidden } else { hidden };
            res_skip_layers.push(WNConv1d::new(
                vb.pp("res_skip_layers").pp(i).pp("conv").pp("conv"),
                hidden,
                out_c,
                1,
                1,
                0,
                1,
                1,
                true,
                None,
                None,
            )?);
        }
        let cond_layer = WNConv1d::new(
            vb.pp("cond_layer").pp("conv").pp("conv"),
            gin,
            2 * hidden * n_layers,
            1,
            1,
            0,
            1,
            1,
            true,
            None,
            None,
        )?;
        Ok(Self {
            in_layers,
            res_skip_layers,
            cond_layer,
            hidden,
            n_layers,
        })
    }

    /// `xs`: (b, hidden, T)；`g`: (b, gin, 1)
    fn forward(&self, xs: &Tensor, g: &Tensor) -> Result<Tensor> {
        let g = self.cond_layer.forward(g)?;
        let mut output = xs.zeros_like()?;
        let mut x = xs.clone();
        let c = self.hidden;
        for i in 0..self.n_layers {
            let x_in = self.in_layers[i].forward(&x)?;
            let g_l = g.narrow(1, i * 2 * c, 2 * c)?;
            // fused_add_tanh_sigmoid_multiply: tanh(前半) * sigmoid(后半)
            let act = x_in.broadcast_add(&g_l)?;
            let t_act = act.narrow(1, 0, c)?.tanh()?;
            let s_act = candle_nn::ops::sigmoid(&act.narrow(1, c, c)?)?;
            let acts = (t_act * s_act)?;
            let res_skip = self.res_skip_layers[i].forward(&acts)?;
            if i < self.n_layers - 1 {
                let res = res_skip.narrow(1, 0, c)?;
                x = (x + res)?;
                output = (output + res_skip.narrow(1, c, c)?)?;
            } else {
                output = (output + res_skip)?;
            }
        }
        Ok(output)
    }
}

/// U-DiT（uvit skip + AdaLN + wavenet 尾）
pub struct Dit {
    blocks: Vec<DitBlock>,
    norm: AdaptiveLayerNorm,
    cond_projection: Linear,
    t_embedder: TimestepEmbedder,
    t_embedder2: TimestepEmbedder,
    conv1: Linear,
    conv2: Conv1d,
    wavenet: Wavenet,
    final_layer: DitFinalLayer,
    res_projection: Linear,
    skip_linear: Linear,
    cond_x_merge_linear: Linear,
    in_channels: usize,
    emit_skip: Vec<usize>,
    receive_skip: Vec<usize>,
    cos: Tensor,
    sin: Tensor,
    device: Device,
}

impl Dit {
    pub fn new(
        vb: &VarBuilder,
        in_channels: usize,
        dim: usize,
        n_head: usize,
        n_layers: usize,
        content_dim: usize,
        wavenet_layers: usize,
    ) -> Result<Self> {
        let head_dim = dim / n_head;
        let intermediate = {
            let mut hidden = 4 * dim;
            hidden = 2 * hidden / 3;
            hidden + (256 - hidden % 256) % 256
        };
        let vb_blocks = vb.pp("transformer").pp("layers");
        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            blocks.push(DitBlock::new(
                &vb_blocks.pp(i),
                dim,
                n_head,
                head_dim,
                intermediate,
            )?);
        }
        let half = n_layers / 2;
        let emit_skip: Vec<usize> = (0..n_layers).filter(|i| *i < half).collect();
        let receive_skip: Vec<usize> = (0..n_layers).filter(|i| *i > half).collect();

        let device = vb.device().clone();
        let dtype = vb.dtype();
        let (cos, sin) = precompute_rope(16384, head_dim, &device, dtype)?;
        // 输入拼接维度从权重反推：dim + 80 + 80 + 192(style)
        let merge_in = vb.get_unchecked("cond_x_merge_linear.weight")?.dim(1)?;
        let skip_in = vb.get_unchecked("skip_linear.weight")?.dim(1)?;

        Ok(Self {
            blocks,
            norm: AdaptiveLayerNorm::new_rms(&vb.pp("transformer").pp("norm"), dim, 1e-5)?,
            cond_projection: linear(content_dim, dim, vb.pp("cond_projection"))?,
            t_embedder: TimestepEmbedder::new(&vb.pp("t_embedder"), dim)?,
            t_embedder2: TimestepEmbedder::new(&vb.pp("t_embedder2"), dim)?,
            conv1: linear(dim, dim, vb.pp("conv1"))?,
            conv2: {
                let weight = vb.get((in_channels, dim, 1), "conv2.weight")?;
                let bias = vb.get(in_channels, "conv2.bias")?;
                Conv1d::new(weight, Some(bias), candle_nn::Conv1dConfig::default())
            },
            wavenet: Wavenet::new(&vb.pp("wavenet"), dim, 5, wavenet_layers, dim)?,
            final_layer: DitFinalLayer::new(&vb.pp("final_layer"), dim)?,
            res_projection: linear(dim, dim, vb.pp("res_projection"))?,
            skip_linear: linear(skip_in, dim, vb.pp("skip_linear"))?,
            cond_x_merge_linear: linear(merge_in, dim, vb.pp("cond_x_merge_linear"))?,
            in_channels,
            emit_skip,
            receive_skip,
            cos,
            sin,
            device,
        })
    }

    /// `x`: (b,80,T) 噪声；`prompt_x`: (b,80,T)；`t`: (b,)；`style`: (b,192)；`cond`: (b,T,512)
    pub fn forward(
        &mut self,
        x: &Tensor,
        prompt_x: &Tensor,
        t: &Tensor,
        style: &Tensor,
        cond: &Tensor,
    ) -> Result<Tensor> {
        let (b, _, frames) = x.dims3()?;
        let t1 = self.t_embedder.forward(t)?; // (b,dim)
        let cond = self.cond_projection.forward(cond)?;

        let x_t = x.transpose(1, 2)?.contiguous()?; // (b,T,80)
        let prompt_t = prompt_x.transpose(1, 2)?.contiguous()?;
        let mut x_in = Tensor::cat(&[&x_t, &prompt_t, &cond], D::Minus1)?; // 672
        let style_rep = style
            .unsqueeze(1)?
            .broadcast_as((b, frames, style.dim(D::Minus1)?))?;
        x_in = Tensor::cat(&[&x_in, &style_rep], D::Minus1)?; // 864
        let mut xs = self.cond_x_merge_linear.forward(&x_in.contiguous()?)?; // (b,T,dim)

        let c = t1.unsqueeze(1)?; // (b,1,dim)
        let cos = self.cos.narrow(0, 0, frames)?.contiguous()?;
        let sin = self.sin.narrow(0, 0, frames)?.contiguous()?;

        let mut skip_pool: Vec<Tensor> = Vec::new();
        for (i, block) in self.blocks.iter_mut().enumerate() {
            let skip_in = if self.receive_skip.contains(&i) {
                skip_pool.pop()
            } else {
                None
            };
            xs = block.forward(&xs, &c, &cos, &sin, skip_in.as_ref())?;
            if self.emit_skip.contains(&i) {
                skip_pool.push(xs.clone());
            }
        }
        let x_res = self.norm.forward(&xs, Some(&c))?;

        // long skip：和输入噪声拼起来
        let x_res = self
            .skip_linear
            .forward(&Tensor::cat(&[&x_res, &x_t], D::Minus1)?)?;

        // wavenet 尾
        let x_mid = self.conv1.forward(&x_res)?; // (b,T,dim)
        let t2 = self.t_embedder2.forward(t)?; // (b,dim)
        let x_w = self
            .wavenet
            .forward(&x_mid.transpose(1, 2)?.contiguous()?, &t2.unsqueeze(2)?)?;
        let x_w = (x_w.transpose(1, 2)? + self.res_projection.forward(&x_res)?)?;
        let x_out = self.final_layer.forward(&x_w, &t1)?; // (b,T,dim)
        let mel = self.conv2.forward(&x_out.transpose(1, 2)?.contiguous()?)?;
        let _ = self.device.clone();
        Ok(mel)
    }
}

/// 上采样长度调节：`content_in_proj` → nearest 插值 → 4×(conv3/GN/Mish) → conv1
pub struct LengthRegulator {
    content_in_proj: Linear,
    model: Vec<(Conv1d, candle_nn::GroupNorm)>,
    proj: Conv1d,
}

impl LengthRegulator {
    pub fn new(
        vb: &VarBuilder,
        in_channels: usize,
        channels: usize,
        n_blocks: usize,
    ) -> Result<Self> {
        let content_in_proj = linear(in_channels, channels, vb.pp("content_in_proj"))?;
        let mut model = Vec::with_capacity(n_blocks);
        
        for i in 0..n_blocks {
            let conv = {
                let weight = vb.get((channels, channels, 3), &format!("model.{}.weight", 3 * i))?;
                let bias = vb.get(channels, &format!("model.{}.bias", 3 * i))?;
                let c = candle_nn::Conv1dConfig {
                    padding: 1,
                    ..Default::default()
                };
                Conv1d::new(weight, Some(bias), c)
            };
            let gn = group_norm(1, channels, 1e-5, vb.pp("model").pp(3 * i + 1))?;
            model.push((conv, gn));
        }
        let proj = {
            let weight = vb.get((channels, channels, 1), "model.12.weight")?;
            let bias = vb.get(channels, "model.12.bias")?;
            Conv1d::new(weight, Some(bias), candle_nn::Conv1dConfig::default())
        };
        Ok(Self {
            content_in_proj,
            model,
            proj,
        })
    }

    /// `xs`: (b,T,in_channels)；`target_len`: 输出帧数
    pub fn forward(&self, xs: &Tensor, target_len: usize) -> Result<Tensor> {
        let xs = self.content_in_proj.forward(xs)?; // (b,T,C)
        let xs = xs.transpose(1, 2)?.contiguous()?; // (b,C,T)
        let xs = nearest_interp(&xs, target_len)?; // (b,C,target)
        let mut xs = xs;
        for (conv, gn) in &self.model {
            xs = conv.forward(&xs)?;
            xs = gn.forward(&xs)?;
            xs = mish(&xs)?;
        }
        let xs = self.proj.forward(&xs)?;
        Ok(xs.transpose(1, 2)?.contiguous()?)
    }
}

/// F.interpolate(mode='nearest')
fn nearest_interp(xs: &Tensor, target: usize) -> Result<Tensor> {
    let (b, c, t) = xs.dims3()?;
    if t == target {
        return Ok(xs.clone());
    }
    let idx: Vec<u32> = (0..target)
        .map(|i| ((i as f64) * (t as f64) / (target as f64)).floor() as u32)
        .map(|i| i.min(t as u32 - 1))
        .collect();
    let idx = Tensor::from_vec(idx, target, xs.device())?;
    let idx = idx.unsqueeze(0)?.broadcast_as((b, target))?.contiguous()?;
    let xs = xs.transpose(1, 2)?.contiguous()?; // (b,T,C)
    let idx = idx
        .unsqueeze(D::Minus1)?
        .broadcast_as((b, target, c))?
        .contiguous()?;
    let out = xs.gather(&idx, 1)?;
    Ok(out.transpose(1, 2)?.contiguous()?)
}

/// CFM 采样器（Euler + CFG）
pub struct Cfm {
    pub estimator: Dit,
}

impl Cfm {
    pub fn new(estimator: Dit) -> Self {
        Self { estimator }
    }

    /// `mu`: (b,T,512)；`prompt`: (b,80,T)；`style`: (b,192)
    /// 返回 (b,80,T) mel
    #[allow(clippy::too_many_arguments)]
    pub fn solve_euler(
        &mut self,
        mu: &Tensor,
        prompt: &Tensor,
        style: &Tensor,
        n_timesteps: usize,
        cfg_rate: f64,
        temperature: f64,
    ) -> Result<Tensor> {
        let (b, t, _) = mu.dims3()?;
        let device = mu.device();
        let dtype = mu.dtype(); // 和 cond 一致
        let prompt_len = prompt.dim(D::Minus1)?;
        let mut x = Tensor::randn(
            0f32,
            temperature as f32,
            (b, self.estimator.in_channels, t),
            device,
        )?
        .to_dtype(dtype)?;
        let mut prompt_x = x.zeros_like()?;
        if prompt_len > 0 {
            prompt_x = Tensor::cat(
                &[
                    prompt.narrow(D::Minus1, 0, prompt_len)?,
                    prompt_x.narrow(D::Minus1, prompt_len, t - prompt_len)?,
                ],
                D::Minus1,
            )?;
            x = Tensor::cat(
                &[
                    x.narrow(D::Minus1, 0, prompt_len)?.zeros_like()?,
                    x.narrow(D::Minus1, prompt_len, t - prompt_len)?,
                ],
                D::Minus1,
            )?;
        }
        let t_span: Vec<f64> = (0..=n_timesteps)
            .map(|i| i as f64 / n_timesteps as f64)
            .collect();
        for step in 1..=n_timesteps {
            let dt = t_span[step] - t_span[step - 1];
            let t_now = Tensor::from_vec(vec![t_span[step - 1] as f32; b], (b,), device)?
                .to_dtype(dtype)?;
            let dphi = if cfg_rate > 0.0 {
                let x2 = Tensor::cat(&[&x, &x], 0)?;
                let p2 = Tensor::cat(&[&prompt_x, &prompt_x.zeros_like()?], 0)?;
                let s2 = Tensor::cat(&[style, &style.zeros_like()?], 0)?;
                let m2 = Tensor::cat(&[mu, &mu.zeros_like()?], 0)?;
                let t2 = Tensor::cat(&[&t_now, &t_now], 0)?;
                let out = self.estimator.forward(&x2, &p2, &t2, &s2, &m2)?;
                let d = out.narrow(0, 0, b)?;
                let d_cfg = out.narrow(0, b, b)?;
                ((d * (1.0 + cfg_rate))? - (d_cfg * cfg_rate)?)?
            } else {
                self.estimator.forward(&x, &prompt_x, &t_now, style, mu)?
            };
            x = (x + (dphi * dt)?)?;
            if prompt_len > 0 {
                x = Tensor::cat(
                    &[
                        x.narrow(D::Minus1, 0, prompt_len)?.zeros_like()?,
                        x.narrow(D::Minus1, prompt_len, t - prompt_len)?,
                    ],
                    D::Minus1,
                )?;
            }
        }
        Ok(x)
    }
}

/// 整体 s2mel：gpt_layer + length_regulator + cfm
pub struct S2Mel {
    pub gpt_layer: Vec<Linear>,
    pub length_regulator: LengthRegulator,
    pub cfm: Cfm,
}

impl S2Mel {
    pub fn new(
        vb: &VarBuilder,
        in_channels: usize,
        dim: usize,
        n_head: usize,
        n_layers: usize,
        content_dim: usize,
        lr_in: usize,
        lr_channels: usize,
        lr_blocks: usize,
        wavenet_layers: usize,
        gpt_dim: usize,
    ) -> Result<Self> {
        let gpt_layer = vec![
            linear(gpt_dim, 256, vb.pp("gpt_layer").pp(0))?,
            linear(256, 128, vb.pp("gpt_layer").pp(1))?,
            // 输出回到语义特征维度（= length_regulator 的输入维度）
            linear(128, lr_in, vb.pp("gpt_layer").pp(2))?,
        ];
        let length_regulator =
            LengthRegulator::new(&vb.pp("length_regulator"), lr_in, lr_channels, lr_blocks)?;
        let dit = Dit::new(
            &vb.pp("cfm").pp("estimator"),
            in_channels,
            dim,
            n_head,
            n_layers,
            content_dim,
            wavenet_layers,
        )?;
        Ok(Self {
            gpt_layer,
            length_regulator,
            cfm: Cfm::new(dit),
        })
    }

    /// GPT hidden (b,T,gpt_dim) → (b,T,content_dim)
    pub fn project_gpt(&self, xs: &Tensor) -> Result<Tensor> {
        let mut h = xs.clone();
        for lin in &self.gpt_layer {
            h = lin.forward(&h)?;
        }
        Ok(h)
    }
}