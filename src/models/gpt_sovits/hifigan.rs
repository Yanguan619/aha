//! SoVITS 的 HiFiGAN 声码器（`module/models.py::Generator` + `ResBlock1`）。
//!
//! v2ProPlus 配置：`initial_channel=192`、`upsample_initial_channel=768`、
//! `upsample_rates=[10,8,2,2,2]`、`upsample_kernels=[20,16,8,2,2]`、`gin_channels=1024`。
//! 192 → 768 → … → 24 → 1，总上采样 640 倍（50 Hz 语义帧 → 32 kHz）。

use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, VarBuilder};

const LRELU_SLOPE: f64 = 0.1;

fn leaky(x: &Tensor, slope: f64) -> Result<Tensor> {
    Ok(candle_nn::ops::leaky_relu(x, slope)?)
}

/// `get_padding(k, d) = (k*d - d) / 2`
fn get_padding(kernel: usize, dilation: usize) -> usize {
    (kernel * dilation - dilation) / 2
}

struct ResBlock1 {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
}

impl ResBlock1 {
    fn new(channels: usize, kernel: usize, dilations: [usize; 3], vb: VarBuilder) -> Result<Self> {
        let mut convs1 = Vec::with_capacity(3);
        let mut convs2 = Vec::with_capacity(3);
        for i in 0..3 {
            convs1.push(candle_nn::conv1d(
                channels,
                channels,
                kernel,
                Conv1dConfig {
                    padding: get_padding(kernel, dilations[i]),
                    dilation: dilations[i],
                    ..Default::default()
                },
                vb.pp(format!("convs1.{}", i)),
            )?);
            convs2.push(candle_nn::conv1d(
                channels,
                channels,
                kernel,
                Conv1dConfig {
                    padding: get_padding(kernel, 1),
                    ..Default::default()
                },
                vb.pp(format!("convs2.{}", i)),
            )?);
        }
        Ok(Self { convs1, convs2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for i in 0..3 {
            let xt = leaky(&x, LRELU_SLOPE)?;
            let xt = self.convs1[i].forward(&xt)?;
            let xt = leaky(&xt, LRELU_SLOPE)?;
            let xt = self.convs2[i].forward(&xt)?;
            x = (&xt + x)?;
        }
        Ok(x)
    }
}

pub struct HifiGan {
    conv_pre: Conv1d,
    cond: Option<Conv1d>,
    ups: Vec<ConvTranspose1d>,
    resblocks: Vec<Vec<ResBlock1>>,
    conv_post: Conv1d,
}

impl HifiGan {
    pub fn load(
        inter_channels: usize,
        upsample_initial_channel: usize,
        upsample_rates: &[usize],
        upsample_kernels: &[usize],
        resblock_kernels: &[usize],
        resblock_dilations: &[[usize; 3]],
        gin_channels: usize,
        conv_post_bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv_pre = candle_nn::conv1d(
            inter_channels,
            upsample_initial_channel,
            7,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb.pp("conv_pre"),
        )?;

        let cond = if gin_channels > 0 {
            Some(candle_nn::conv1d(
                gin_channels,
                upsample_initial_channel,
                1,
                Conv1dConfig::default(),
                vb.pp("cond"),
            )?)
        } else {
            None
        };

        let mut ups = Vec::with_capacity(upsample_rates.len());
        let mut resblocks = Vec::with_capacity(upsample_rates.len());
        for i in 0..upsample_rates.len() {
            let u = upsample_rates[i];
            let k = upsample_kernels[i];
            let in_c = upsample_initial_channel / (1 << i);
            let out_c = upsample_initial_channel / (1 << (i + 1));
            ups.push(candle_nn::conv_transpose1d(
                in_c,
                out_c,
                k,
                ConvTranspose1dConfig {
                    stride: u,
                    padding: (k - u) / 2,
                    ..Default::default()
                },
                vb.pp(format!("ups.{}", i)),
            )?);

            let mut stage = Vec::with_capacity(resblock_kernels.len());
            for j in 0..resblock_kernels.len() {
                stage.push(ResBlock1::new(
                    out_c,
                    resblock_kernels[j],
                    resblock_dilations[j],
                    vb.pp(format!("resblocks.{}", i * resblock_kernels.len() + j)),
                )?);
            }
            resblocks.push(stage);
        }

        let final_channels = upsample_initial_channel / (1 << upsample_rates.len());
        let conv_post = if conv_post_bias {
            candle_nn::conv1d(
                final_channels,
                1,
                7,
                Conv1dConfig {
                    padding: 3,
                    ..Default::default()
                },
                vb.pp("conv_post"),
            )?
        } else {
            candle_nn::conv1d_no_bias(
                final_channels,
                1,
                7,
                Conv1dConfig {
                    padding: 3,
                    ..Default::default()
                },
                vb.pp("conv_post"),
            )?
        };

        Ok(Self {
            conv_pre,
            cond,
            ups,
            resblocks,
            conv_post,
        })
    }

    fn debug(&self) -> bool {
        std::env::var("AHA_GSV_DEBUG").is_ok()
    }

    fn log_stage(&self, label: &str, x: &Tensor) -> Result<()> {
        if self.debug() {
            let v: Vec<f32> = x.flatten_all()?.to_vec1()?;
            let n = v.len() as f64;
            let mean = v.iter().map(|a| *a as f64).sum::<f64>() / n;
            let std = (v.iter().map(|a| (*a as f64 - mean).powi(2)).sum::<f64>() / n).sqrt();
            let peak = v.iter().fold(0f32, |m, a| m.max(a.abs()));
            eprintln!("    [dec] {label:<14} dims={:?} std={std:.4} peak={peak:.4}", x.dims());
        }
        Ok(())
    }

    /// `x`: (B, 192, T)；`g`: (B, 1024, 1) → (B, 1, N) 波形
    pub fn forward(&self, x: &Tensor, g: Option<&Tensor>) -> Result<Tensor> {
        self.log_stage("input", x)?;
        let mut x = self.conv_pre.forward(x)?;
        self.log_stage("conv_pre", &x)?;
        if let (Some(cond), Some(g)) = (&self.cond, g) {
            // g 只有 1 帧，广播到全部时间步
            x = x.broadcast_add(&cond.forward(g)?)?;
            self.log_stage("+cond", &x)?;
        }

        for i in 0..self.ups.len() {
            x = leaky(&x, LRELU_SLOPE)?;
            x = self.ups[i].forward(&x)?;
            let mut sum: Option<Tensor> = None;
            for rb in &self.resblocks[i] {
                let y = rb.forward(&x)?;
                sum = Some(match sum {
                    None => y,
                    Some(acc) => (acc + y)?,
                });
            }
            let n = self.resblocks[i].len() as f64;
            x = (sum.expect("resblocks 非空") / n)?;
            self.log_stage(&format!("stage{i}"), &x)?;
        }

        // 最后一次 leaky_relu 用的是默认斜率 0.01
        x = leaky(&x, 0.01)?;
        self.log_stage("pre_post", &x)?;
        x = self.conv_post.forward(&x)?;
        self.log_stage("conv_post", &x)?;
        Ok(x.tanh()?)
    }
}
