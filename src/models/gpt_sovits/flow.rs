//! SoVITS 的 flow（`module/modules.py` 的 `WN` + `ResidualCouplingLayer`）。
//!
//! 结构：4 × [ResidualCouplingLayer + Flip]，推理走 `reverse=True`：
//!
//! ```text
//! for flow in reversed(flows):
//!     Flip:  x = flip(x, dim=1)
//!     RCL:   x1' = (x1 - m) * exp(-logs) * mask      # mean_only=True → logs = 0
//!            m = post(WN(pre(x0) * mask, g)) * mask
//! ```
//!
//! 注意：`mean_only=True` 时 `post` 只输出 `half_channels` 维，且 `logs` 恒为 0。

use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, VarBuilder, conv1d};

/// WaveNet（`module/modules.py::WN`），带全局条件 `g`
struct Wn {
    in_layers: Vec<Conv1d>,
    res_skip_layers: Vec<Conv1d>,
    cond_layer: Option<Conv1d>,
    hidden: usize,
    n_layers: usize,
}

impl Wn {
    fn new(
        hidden: usize,
        kernel: usize,
        dilation_rate: usize,
        n_layers: usize,
        gin_channels: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut in_layers = Vec::with_capacity(n_layers);
        let mut res_skip_layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let dilation = dilation_rate.pow(i as u32);
            let padding = (kernel * dilation - dilation) / 2;
            in_layers.push(conv1d(
                hidden,
                hidden * 2,
                kernel,
                Conv1dConfig {
                    padding,
                    dilation,
                    ..Default::default()
                },
                vb.pp(format!("in_layers.{}", i)),
            )?);
            // 最后一层的 res_skip 只输出 hidden（直接整体加到 output 上）
            let res_skip_channels = if i < n_layers - 1 { hidden * 2 } else { hidden };
            res_skip_layers.push(conv1d(
                hidden,
                res_skip_channels,
                1,
                Conv1dConfig::default(),
                vb.pp(format!("res_skip_layers.{}", i)),
            )?);
        }
        let cond_layer = if gin_channels > 0 {
            Some(conv1d(
                gin_channels,
                hidden * 2 * n_layers,
                1,
                Conv1dConfig::default(),
                vb.pp("cond_layer"),
            )?)
        } else {
            None
        };
        Ok(Self {
            in_layers,
            res_skip_layers,
            cond_layer,
            hidden,
            n_layers,
        })
    }

    /// `x`: (B, C, T)；`g`: (B, gin, 1)
    fn forward(&self, x: &Tensor, g: Option<&Tensor>) -> Result<Tensor> {
        let mut x = x.clone();
        let mut output = x.zeros_like()?;
        let g = match (g, &self.cond_layer) {
            (Some(g), Some(layer)) => Some(layer.forward(g)?),
            _ => None,
        };

        for i in 0..self.n_layers {
            let x_in = self.in_layers[i].forward(&x)?;
            let acts = match &g {
                Some(g) => {
                    let offset = i * 2 * self.hidden;
                    let g_l = g.narrow(1, offset, 2 * self.hidden)?;
                    fused_add_tanh_sigmoid(&x_in, &g_l, self.hidden)?
                }
                None => {
                    let h = self.hidden;
                    let a = x_in.narrow(1, 0, h)?.tanh()?;
                    let b = candle_nn::ops::sigmoid(&x_in.narrow(1, h, h)?)?;
                    (a * b)?
                }
            };
            let res_skip = self.res_skip_layers[i].forward(&acts)?;
            if i < self.n_layers - 1 {
                let res = res_skip.narrow(1, 0, self.hidden)?;
                x = (x + res)?;
                output = (&output + res_skip.narrow(1, self.hidden, self.hidden)?)?;
            } else {
                output = (&output + res_skip)?;
            }
        }
        Ok(output)
    }
}

fn fused_add_tanh_sigmoid(a: &Tensor, b: &Tensor, channels: usize) -> Result<Tensor> {
    // g 只有 1 帧，需要广播到时间维
    let sum = a.broadcast_add(b)?;
    let t = sum.narrow(1, 0, channels)?.tanh()?;
    let s = candle_nn::ops::sigmoid(&sum.narrow(1, channels, channels)?)?;
    Ok((t * s)?)
}

struct ResidualCouplingLayer {
    pre: Conv1d,
    enc: Wn,
    post: Conv1d,
    half: usize,
}

impl ResidualCouplingLayer {
    fn new(
        channels: usize,
        hidden: usize,
        kernel: usize,
        dilation_rate: usize,
        n_layers: usize,
        gin_channels: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let half = channels / 2;
        Ok(Self {
            pre: conv1d(half, hidden, 1, Conv1dConfig::default(), vb.pp("pre"))?,
            enc: Wn::new(hidden, kernel, dilation_rate, n_layers, gin_channels, vb.pp("enc"))?,
            post: conv1d(hidden, half, 1, Conv1dConfig::default(), vb.pp("post"))?,
            half,
        })
    }

    /// `reverse=true`：`x1 = (x1 - m) * exp(-logs)`
    fn reverse(&self, x: &Tensor, g: Option<&Tensor>) -> Result<Tensor> {
        let x0 = x.narrow(1, 0, self.half)?;
        let x1 = x.narrow(1, self.half, self.half)?;
        let h = self.pre.forward(&x0)?;
        let h = self.enc.forward(&h, g)?;
        let m = self.post.forward(&h)?;
        let x1 = (x1 - m)?;
        Ok(Tensor::cat(&[&x0, &x1], 1)?)
    }
}

enum Flow {
    Coupling(Box<ResidualCouplingLayer>),
    Flip,
}

/// `ResidualCouplingBlock`（推理只用 reverse）
pub struct FlowBlock {
    flows: Vec<Flow>,
    gin_channels: usize,
}

impl FlowBlock {
    pub fn load(
        channels: usize,
        hidden: usize,
        kernel: usize,
        dilation_rate: usize,
        n_layers: usize,
        n_flows: usize,
        gin_channels: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut flows = Vec::with_capacity(n_flows * 2);
        for i in 0..n_flows {
            flows.push(Flow::Coupling(Box::new(ResidualCouplingLayer::new(
                channels,
                hidden,
                kernel,
                dilation_rate,
                n_layers,
                gin_channels,
                vb.pp(format!("flows.{}", i * 2)),
            )?)));
            flows.push(Flow::Flip);
        }
        Ok(Self {
            flows,
            gin_channels,
        })
    }

    /// `x`: (B, C, T)；`g`: 音色向量 (B, gin, 1)
    pub fn reverse(&self, x: &Tensor, g: Option<&Tensor>) -> Result<Tensor> {
        if g.is_some() && self.gin_channels == 0 {
            anyhow::bail!("flow 没有条件通道，但传入了 g");
        }
        let mut x = x.clone();
        for flow in self.flows.iter().rev() {
            x = match flow {
                // torch.flip(x, [1])
                Flow::Flip => {
                    let c = x.dim(1)?;
                    let idx = Tensor::from_vec(
                        (0..c as u32).rev().collect::<Vec<_>>(),
                        c,
                        x.device(),
                    )?;
                    x.index_select(&idx, 1)?
                }
                Flow::Coupling(layer) => layer.reverse(&x, g)?,
            };
        }
        Ok(x)
    }
}
