//! ERes2NetV2 说话人向量模型（`eres2net/ERes2NetV2.py`，v2Pro 的 `sv`）。
//!
//! `ERes2NetV2(baseWidth=24, scale=4, expansion=4)`，输入 80 维 kaldi fbank `(T, 80)`，
//! `forward3` 输出 20480 维（2048 通道 × 10 频带，时间维取均值）。
//!
//! ```text
//! fbank(T,80) → (1,1,80,T) → conv1(64,k3) → layer1(3×block) → layer2(4×block,s2)
//!   → layer3(6×block_fuse,s2) → layer4(3×block_fuse,s2)
//!   → layer3_ds(1024→2048,k3,s2) → fuse34(AFF)
//!   → (1,2048,10,T/8) → 时间维均值 → (1, 20480)
//! ```
//!
//! 注意 `ReLU` 是 `Hardtanh(0, 20)`（不是普通 relu），`fuse_models` 只在
//! `block_fuse` 里出现。

use std::path::Path;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{BatchNorm, Conv2d, Conv2dConfig, Module, ModuleT, VarBuilder, batch_norm, conv2d, conv2d_no_bias};

const BASE_WIDTH: usize = 24;
const SCALE: usize = 4;
const EXPANSION: usize = 4;
const M_CHANNELS: usize = 64;

/// `Hardtanh(0, 20)`（源码里名字叫 ReLU，但不是 relu）
fn hard_relu(x: &Tensor) -> Result<Tensor> {
    Ok(x.clamp(0f32, 20f32)?)
}

/// AFF 注意力融合（`fusion.py::AFF`）
struct Aff {
    conv_a: Conv2d,
    bn_a: BatchNorm,
    conv_b: Conv2d,
    bn_b: BatchNorm,
}

impl Aff {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let inter = channels / 4;
        let cfg = Conv2dConfig::default();
        Ok(Self {
            conv_a: conv2d(channels * 2, inter, 1, cfg, vb.pp("local_att.0"))?,
            bn_a: batch_norm(inter, 1e-5, vb.pp("local_att.1"))?,
            conv_b: conv2d(inter, channels, 1, cfg, vb.pp("local_att.3"))?,
            bn_b: batch_norm(channels, 1e-5, vb.pp("local_att.4"))?,
        })
    }

    /// `x_att = 1 + tanh(local_att(cat(x, ds_y)))`；`x*x_att + ds_y*(2-x_att)`
    fn forward(&self, x: &Tensor, ds_y: &Tensor) -> Result<Tensor> {
        let xa = Tensor::cat(&[x, ds_y], 1)?;
        let h = self.conv_a.forward(&xa)?;
        let h = self.bn_a.forward_t(&h, false)?;
        let h = candle_nn::ops::silu(&h)?;
        let h = self.conv_b.forward(&h)?;
        let h = self.bn_b.forward_t(&h, false)?;
        // x_att = 1 + tanh(h)
        let x_att = h.tanh()?.affine(1.0, 1.0)?;
        let left = x.mul(&x_att)?;
        // 2 - x_att
        let two = x_att.affine(-1.0, 2.0)?;
        let right = ds_y.mul(&two)?;
        Ok(left.add(&right)?)
    }
}

/// `BasicBlockERes2NetV2`（无融合版）
struct Block {
    conv1: Conv2d,
    bn1: BatchNorm,
    convs: Vec<Conv2d>,
    bns: Vec<BatchNorm>,
    fuse: Vec<Aff>, // block_fuse 才有
    conv3: Conv2d,
    bn3: BatchNorm,
    shortcut: Option<(Conv2d, BatchNorm)>,
    width: usize,
}

impl Block {
    fn new(
        in_planes: usize,
        planes: usize,
        stride: usize,
        use_fuse: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let width = (planes * BASE_WIDTH) / 64;
        let k1 = Conv2dConfig {
            stride,
            ..Default::default()
        };
        let k3 = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };

        let mut convs = Vec::with_capacity(SCALE);
        let mut bns = Vec::with_capacity(SCALE);
        for i in 0..SCALE {
            convs.push(conv2d_no_bias(width, width, 3, k3, vb.pp(format!("convs.{}", i)))?);
            bns.push(batch_norm(width, 1e-5, vb.pp(format!("bns.{}", i)))?);
        }
        let mut fuse = Vec::new();
        if use_fuse {
            for j in 0..SCALE - 1 {
                fuse.push(Aff::new(width, vb.pp(format!("fuse_models.{}", j)))?);
            }
        }

        let shortcut = if stride != 1 || in_planes != EXPANSION * planes {
            Some((
                conv2d_no_bias(in_planes, EXPANSION * planes, 1, k1, vb.pp("shortcut.0"))?,
                batch_norm(EXPANSION * planes, 1e-5, vb.pp("shortcut.1"))?,
            ))
        } else {
            None
        };

        Ok(Self {
            conv1: conv2d_no_bias(in_planes, width * SCALE, 1, k1, vb.pp("conv1"))?,
            bn1: batch_norm(width * SCALE, 1e-5, vb.pp("bn1"))?,
            convs,
            bns,
            fuse,
            conv3: conv2d_no_bias(width * SCALE, planes * EXPANSION, 1, Conv2dConfig::default(), vb.pp("conv3"))?,
            bn3: batch_norm(planes * EXPANSION, 1e-5, vb.pp("bn3"))?,
            shortcut,
            width,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = self.bn1.forward_t(&self.conv1.forward(x)?, false)?;
        out = hard_relu(&out)?;

        // spx = split(out, width, dim=1)
        let mut spx = Vec::with_capacity(SCALE);
        for i in 0..SCALE {
            spx.push(out.narrow(1, i * self.width, self.width)?);
        }

        let mut sp = spx[0].clone();
        let mut acc: Option<Tensor> = None;
        for i in 0..SCALE {
            let cur = if i == 0 {
                spx[0].clone()
            } else if self.fuse.is_empty() {
                sp.add(&spx[i])?
            } else {
                self.fuse[i - 1].forward(&sp, &spx[i])?
            };
            // sp 跨迭代传递：sp = relu(bn(conv(sp + spx[i])))
            sp = self.convs[i].forward(&cur)?;
            sp = self.bns[i].forward_t(&sp, false)?;
            sp = hard_relu(&sp)?;
            let piece = sp.clone();
            acc = Some(match acc {
                None => piece,
                Some(prev) => Tensor::cat(&[&prev, &piece], 1)?,
            });
        }
        let mut out = self.bn3.forward_t(&self.conv3.forward(&acc.expect("scale>=1"))?, false)?;

        if let Some((conv, bn)) = &self.shortcut {
            let residual = bn.forward_t(&conv.forward(x)?, false)?;
            out = out.add(&residual)?;
        } else {
            out = out.add(x)?;
        }
        hard_relu(&out)
    }
}

fn make_layer(
    vb: VarBuilder,
    mut in_planes: usize,
    planes: usize,
    num_blocks: usize,
    stride: usize,
    use_fuse: bool,
) -> Result<(Vec<Block>, usize)> {
    let mut blocks = Vec::with_capacity(num_blocks);
    for i in 0..num_blocks {
        let s = if i == 0 { stride } else { 1 };
        blocks.push(Block::new(in_planes, planes, s, use_fuse, vb.pp(i))?);
        in_planes = planes * EXPANSION;
    }
    Ok((blocks, in_planes))
}

pub struct ERes2NetV2 {
    device: Device,
    conv1: Conv2d,
    bn1: BatchNorm,
    layer1: Vec<Block>,
    layer2: Vec<Block>,
    layer3: Vec<Block>,
    layer4: Vec<Block>,
    layer3_ds: Conv2d,
    fuse34: Aff,
}

impl ERes2NetV2 {
    pub fn load(dir: impl AsRef<Path>, device: &Device, dtype: DType) -> Result<Self> {
        let tensors = candle_core::safetensors::load(dir.as_ref().join("sv_eres2net.safetensors"), device)?;
        let vb = VarBuilder::from_tensors(tensors, dtype, device);

        let cfg3 = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let (layer1, in1) = make_layer(vb.pp("layer1"), M_CHANNELS, M_CHANNELS, 3, 1, false)?;
        let (layer2, in2) = make_layer(vb.pp("layer2"), in1, M_CHANNELS * 2, 4, 2, false)?;
        let (layer3, in3) = make_layer(vb.pp("layer3"), in2, M_CHANNELS * 4, 6, 2, true)?;
        let (layer4, _) = make_layer(vb.pp("layer4"), in3, M_CHANNELS * 8, 3, 2, true)?;

        Ok(Self {
            device: device.clone(),
            conv1: conv2d_no_bias(1, M_CHANNELS, 3, cfg3, vb.pp("conv1"))?,
            bn1: batch_norm(M_CHANNELS, 1e-5, vb.pp("bn1"))?,
            layer1,
            layer2,
            layer3,
            layer4,
            layer3_ds: conv2d_no_bias(
                M_CHANNELS * 4 * EXPANSION,
                M_CHANNELS * 8 * EXPANSION,
                3,
                Conv2dConfig {
                    padding: 1,
                    stride: 2,
                    ..Default::default()
                },
                vb.pp("layer3_ds"),
            )?,
            fuse34: Aff::new(M_CHANNELS * 8 * EXPANSION, vb.pp("fuse34"))?,
        })
    }

    /// fbank `(T, 80)` → `(1, 20480)`（对应 `sv.compute_embedding3`）
    pub fn forward3(&self, fbank: &Tensor) -> Result<Tensor> {
        // (T,80) -> (1,1,80,T)
        let x = fbank.t()?.unsqueeze(0)?.unsqueeze(0)?.contiguous()?;

        let mut out = self.bn1.forward_t(&self.conv1.forward(&x)?, false)?;
        out = hard_relu(&out)?;

        for b in &self.layer1 {
            out = b.forward(&out)?;
        }
        for b in &self.layer2 {
            out = b.forward(&out)?;
        }
        let out3 = {
            let mut o = out.clone();
            for b in &self.layer3 {
                o = b.forward(&o)?;
            }
            o
        };
        let out4 = {
            let mut o = out3.clone();
            for b in &self.layer4 {
                o = b.forward(&o)?;
            }
            o
        };

        let out3_ds = self.layer3_ds.forward(&out3)?;
        let fused = self.fuse34.forward(&out4, &out3_ds)?;

        // (1, 2048, 10, T') -> flatten(1,2) -> (1, 20480, T') -> mean(-1)
        let (b, c, f, t) = fused.dims4()?;
        let flat = fused.reshape((b, c * f, t))?;
        let mean = flat.mean(candle_core::D::Minus1)?;
        Ok(mean)
    }
}

// ---------------------------------------------------------------------------
// Kaldi fbank（torchaudio.compliance.kaldi.fbank 的等价实现，**不做 32768 缩放**）
// ---------------------------------------------------------------------------

fn hz_to_mel_kaldi(f: f64) -> f64 {
    1127.0 * (1.0 + f / 700.0).ln()
}

fn povey_window(n: usize) -> Vec<f32> {
    let denom = (n - 1) as f64;
    (0..n)
        .map(|i| {
            let v = 2.0 * std::f64::consts::PI * i as f64 / denom;
            ((0.5 - 0.5 * v.cos()).powf(0.85)) as f32
        })
        .collect()
}

fn kaldi_mel_bank(sr: usize, fft_len: usize, num_bins: usize) -> Vec<Vec<f32>> {
    let n_bins = fft_len / 2 + 1;
    let (low, high) = (20.0f64, sr as f64 / 2.0);
    let (mel_low, mel_high) = (hz_to_mel_kaldi(low), hz_to_mel_kaldi(high));
    let centers: Vec<f64> = (0..num_bins + 2)
        .map(|i| mel_low + (mel_high - mel_low) * i as f64 / (num_bins + 1) as f64)
        .collect();
    let freqs_mel: Vec<f64> = (0..n_bins)
        .map(|i| hz_to_mel_kaldi(i as f64 * sr as f64 / fft_len as f64))
        .collect();
    (0..num_bins)
        .map(|b| {
            let (l, c, r) = (centers[b], centers[b + 1], centers[b + 2]);
            freqs_mel
                .iter()
                .map(|m| {
                    if *m <= l || *m >= r {
                        0.0
                    } else if *m <= c {
                        ((*m - l) / (c - l)) as f32
                    } else {
                        ((r - *m) / (r - c)) as f32
                    }
                })
                .collect()
        })
        .collect()
}

/// 简易 FFT（radix-2，n 为 2 的幂）
fn fft_inplace(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    if n <= 1 {
        return;
    }
    let mut j = 0usize;
    for i in 0..n {
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
        let mut m = n >> 1;
        while m >= 1 && j & m != 0 {
            j &= !m;
            m >>= 1;
        }
        j |= m;
    }
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * std::f32::consts::PI / len as f32;
        let (wr, wi) = (ang.cos(), ang.sin());
        let mut i = 0;
        while i < n {
            let (mut cr, mut ci) = (1.0f32, 0.0f32);
            for k in 0..len / 2 {
                let (a, b) = (i + k, i + k + len / 2);
                let (tr, ti) = (re[b] * cr - im[b] * ci, re[b] * ci + im[b] * cr);
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
                let (nr, ni) = (cr * wr - ci * wi, cr * wi + ci * wr);
                cr = nr;
                ci = ni;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// 80 维 kaldi fbank（povey 窗、去直流、0.97 预加重、功率谱、log）→ `(T, 80)`
pub fn sv_fbank(audio_16k: &[f32]) -> Result<Vec<Vec<f32>>> {
    const FRAME: usize = 400;
    const HOP: usize = 160;
    const FFT: usize = 512;
    const EPS: f32 = 1.192_092_9e-7;

    let window = povey_window(FRAME);
    let bank = kaldi_mel_bank(16_000, FFT, 80);
    if audio_16k.len() < FRAME {
        anyhow::bail!("参考音频太短（{} < {FRAME}）", audio_16k.len());
    }
    let frames = (audio_16k.len() - FRAME) / HOP + 1;
    let mut out = Vec::with_capacity(frames);
    let mut re = vec![0f32; FFT];
    let mut im = vec![0f32; FFT];
    for f in 0..frames {
        let start = f * HOP;
        let mut buf = audio_16k[start..start + FRAME].to_vec();
        let mean = buf.iter().sum::<f32>() / FRAME as f32;
        for v in buf.iter_mut() {
            *v -= mean;
        }
        for i in (1..FRAME).rev() {
            buf[i] -= 0.97 * buf[i - 1];
        }
        buf[0] *= 1.0 - 0.97;
        for i in 0..FRAME {
            buf[i] *= window[i];
        }
        for i in 0..FFT {
            re[i] = if i < FRAME { buf[i] } else { 0.0 };
            im[i] = 0.0;
        }
        fft_inplace(&mut re, &mut im);
        let power: Vec<f32> = (0..FFT / 2 + 1).map(|i| re[i] * re[i] + im[i] * im[i]).collect();
        let mut fb = Vec::with_capacity(80);
        for b in bank.iter() {
            let e: f32 = b.iter().zip(power.iter()).map(|(w, p)| w * p).sum();
            fb.push(e.max(EPS).ln());
        }
        out.push(fb);
    }
    Ok(out)
}

impl ERes2NetV2 {
    /// 16 kHz 参考音频 → `(1, 20480)` 说话人向量
    pub fn compute_sv(&self, audio_16k: &[f32]) -> Result<Tensor> {
        let feat = sv_fbank(audio_16k)?;
        let mut data = Vec::with_capacity(feat.len() * 80);
        for row in &feat {
            data.extend_from_slice(row);
        }
        let fbank = Tensor::from_vec(data, (feat.len(), 80), &self.device)?;
        self.forward3(&fbank)
    }
}
