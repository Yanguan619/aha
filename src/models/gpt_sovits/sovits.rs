//! SoVITS 合成器（v2 / v2Pro）。
//!
//! 端到端（对应 `TTS.py` 的 `decode` 阶段）：
//!
//! ```text
//! AR codes(T) ─quantizer.decode─> (1,768,T) ─×2 nearest─> (1,768,2T)
//!   文本 phones ─embedding─> (1,192,S)                     │
//!   ge(1,1024,1) ─ge_to512─> (1,512,1)                     │
//!                          ↓                              ↓
//!   enc_p: ssl_proj(768→192) → encoder_ssl → 与文本做 MRTE → encoder2 → proj
//!          → m, logs → z_p = m + N(0,1)·exp(logs)·noise_scale
//!                          ↓
//!   flow.reverse(z_p, g=ge) → z
//!                          ↓
//!   dec(z, g=ge) → 32 kHz 波形
//! ```

use std::path::Path;

use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Embedding, Linear, VarBuilder, conv1d, embedding, linear};
use serde::Deserialize;

use super::flow::FlowBlock;
use super::hifigan::HifiGan;
use super::eres2net::ERes2NetV2;
use super::quantizer::SemanticQuantizer;
use super::style::{self, StyleEncoder};
use super::vits_attn::{Mrte, VitsEncoder};

/// `manifest.json` 里的 SoVITS 超参（`hps.model` + `audio`）
#[derive(Debug, Clone, Deserialize)]
pub struct SovitsConfig {
    pub inter_channels: usize,
    pub hidden_channels: usize,
    pub filter_channels: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub kernel_size: usize,
    pub resblock_kernel_sizes: Vec<usize>,
    pub resblock_dilation_sizes: Vec<[usize; 3]>,
    pub upsample_rates: Vec<usize>,
    pub upsample_kernel_sizes: Vec<usize>,
    pub upsample_initial_channel: usize,
    pub gin_channels: usize,
    #[serde(default = "default_frame_rate")]
    pub semantic_frame_rate: String,
}

fn default_frame_rate() -> String {
    "25hz".to_string()
}

#[derive(Debug, Deserialize)]
struct Hps {
    model: SovitsConfig,
}

#[derive(Debug, Deserialize)]
struct AudioInfo {
    sampling_rate: usize,
    hop_length: usize,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    hps: Hps,
    audio: AudioInfo,
    #[serde(default)]
    is_v2pro: bool,
}

/// 合成参数
#[derive(Debug, Clone)]
pub struct SynthOptions {
    /// 语速（`1.0` = 原始）
    pub speed: f32,
    /// `decode` 里的 noise_scale，官方 TTS 默认 0.5
    pub noise_scale: f32,
    /// 噪声种子（Rust 侧自定义 RNG，不与 torch 逐位一致）
    pub seed: u64,
}

impl Default for SynthOptions {
    fn default() -> Self {
        Self {
            speed: 1.0,
            noise_scale: 0.5,
            seed: 42,
        }
    }
}

struct EncP {
    ssl_proj: Conv1d,
    encoder_ssl: VitsEncoder,
    text_embedding: Embedding,
    encoder_text: VitsEncoder,
    mrte: Mrte,
    encoder2: VitsEncoder,
    proj: Conv1d,
    out_channels: usize,
}

impl EncP {
    fn load(cfg: &SovitsConfig, vb: VarBuilder) -> Result<Self> {
        let eps = 1e-5;
        let hidden = cfg.hidden_channels;
        let inter = cfg.inter_channels;
        Ok(Self {
            ssl_proj: conv1d(768, hidden, 1, Conv1dConfig::default(), vb.pp("ssl_proj"))?,
            encoder_ssl: VitsEncoder::new(
                hidden,
                cfg.filter_channels,
                cfg.n_heads,
                cfg.n_layers / 2,
                cfg.kernel_size,
                Some(4),
                eps,
                vb.pp("encoder_ssl"),
            )?,
            text_embedding: embedding(732, hidden, vb.pp("text_embedding"))?,
            encoder_text: VitsEncoder::new(
                hidden,
                cfg.filter_channels,
                cfg.n_heads,
                cfg.n_layers,
                cfg.kernel_size,
                Some(4),
                eps,
                vb.pp("encoder_text"),
            )?,
            mrte: Mrte::new(hidden, 512, inter, 4, vb.pp("mrte"))?,
            encoder2: VitsEncoder::new(
                hidden,
                cfg.filter_channels,
                cfg.n_heads,
                cfg.n_layers / 2,
                cfg.kernel_size,
                Some(4),
                eps,
                vb.pp("encoder2"),
            )?,
            proj: conv1d(inter, inter * 2, 1, Conv1dConfig::default(), vb.pp("proj"))?,
            out_channels: inter,
        })
    }

    /// `quantized`: (1, 768, T)；`text`: (S,)；`ge512`: (1, 512, 1)
    fn forward(
        &self,
        quantized: &Tensor,
        text: &[u32],
        ge512: &Tensor,
        speed: f32,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let y = self.ssl_proj.forward(quantized)?; // (1,192,T)
        dump_stage("enc_ssl_proj", &y)?;
        let y = self.encoder_ssl.forward_plain(&y)?;
        dump_stage("enc_ssl", &y)?;

        let t = self.text_embedding.forward(&Tensor::from_vec(
            text.to_vec(),
            (1, text.len()),
            device,
        )?)?;
        let t = t.transpose(1, 2)?.contiguous()?; // (1,192,S)
        dump_stage("enc_text_emb", &t)?;
        let t = self.encoder_text.forward_plain(&t)?;
        dump_stage("enc_text", &t)?;

        let y = self.mrte.forward_plain(&y, &t, ge512)?;
        dump_stage("enc_mrte", &y)?;
        let mut y = self.encoder2.forward_plain(&y)?;
        dump_stage("enc2", &y)?;

        if (speed - 1.0).abs() > f32::EPSILON {
            let target = (y.dim(2)? as f32 / speed) as usize + 1;
            y = interpolate_linear(&y, target)?;
        }

        let stats = self.proj.forward(&y)?; // (1, 384, T)
        dump_stage("enc_proj", &stats)?;
        let m = stats.narrow(1, 0, self.out_channels)?;
        let logs = stats.narrow(1, self.out_channels, self.out_channels)?;
        Ok((m, logs))
    }
}

/// 诊断用：把张量写成裸 f32 到 `<model_dir>/ref_dump/rust_<name>.bin`
fn dump_stage(name: &str, t: &Tensor) -> Result<()> {
    if std::env::var("AHA_GSV_DUMP").is_err() {
        return Ok(());
    }
    let Some(dir) = std::env::var("AHA_GPT_SOVITS_DIR").ok() else {
        return Ok(());
    };
    let dir = std::path::Path::new(&dir).join("ref_dump");
    std::fs::create_dir_all(&dir)?;
    let v: Vec<f32> = t.flatten_all()?.to_vec1()?;
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    std::fs::write(dir.join(format!("rust_{name}.bin")), bytes)?;
    Ok(())
}

/// `F.interpolate(mode="linear")`（沿时间维，单样本）
fn interpolate_linear(x: &Tensor, target: usize) -> Result<Tensor> {
    let t = x.dim(2)?;
    if target == t {
        return Ok(x.clone());
    }
    let scale = if target > 1 {
        (t as f32 - 1.0) / (target as f32 - 1.0)
    } else {
        0.0
    };
    let mut frames = Vec::with_capacity(target);
    for i in 0..target {
        let pos = i as f32 * scale;
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(t - 1);
        let w = pos - lo as f32;
        let a = x.narrow(2, lo, 1)?;
        let b = x.narrow(2, hi, 1)?;
        frames.push(((a * (1.0 - w) as f64)? + (b * w as f64)?)?);
    }
    Ok(Tensor::cat(&frames, 2)?.contiguous()?)
}

/// 最近邻 ×2（`F.interpolate(mode="nearest")`）
fn upsample_x2(x: &Tensor) -> Result<Tensor> {
    let (b, c, t) = x.dims3()?;
    Ok(x
        .unsqueeze(3)?
        .broadcast_as((b, c, t, 2))?
        .reshape((b, c, t * 2))?
        .contiguous()?)
}

/// 简易 RNG（xorshift64*），种子固定即可复现；不和 torch 逐位一致。
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / (1u64 << 24) as f32
    }

    /// Box-Muller
    fn next_normal(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-7);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

pub struct Sovits {
    sv: Option<ERes2NetV2>,
    quantizer: SemanticQuantizer,
    style: StyleEncoder,
    enc_p: EncP,
    flow: FlowBlock,
    dec: HifiGan,
    ge_to512: Linear,
    cfg: SovitsConfig,
    sampling_rate: usize,
    hop_length: usize,
    is_v2pro: bool,
    device: Device,
}

impl Sovits {
    pub fn load(dir: impl AsRef<Path>, device: &Device, dtype: DType) -> Result<Self> {
        let dir = dir.as_ref();
        let manifest: Manifest = serde_json::from_reader(std::io::BufReader::new(
            std::fs::File::open(dir.join("manifest.json"))?,
        ))?;
        let cfg = manifest.hps.model;
        let is_v2pro = manifest.is_v2pro || style::is_v2pro(dir)?;

        let tensors = candle_core::safetensors::load(dir.join("sovits.safetensors"), device)?;
        let vb = VarBuilder::from_tensors(tensors, dtype, device);

        let flow = FlowBlock::load(
            cfg.inter_channels,
            cfg.inter_channels,
            5, // SynthesizerTrn 里 flow 的 kernel_size 固定为 5（与 enc_p 的 kernel 不同）
            1,
            4,
            4,
            cfg.gin_channels,
            vb.pp("flow"),
        )?;
        let dec = HifiGan::load(
            cfg.inter_channels,
            cfg.upsample_initial_channel,
            &cfg.upsample_rates,
            &cfg.upsample_kernel_sizes,
            &cfg.resblock_kernel_sizes,
            &cfg.resblock_dilation_sizes,
            cfg.gin_channels,
            false, // conv_post 无 bias（与权重一致）
            vb.pp("dec"),
        )?;

        let enc_p = EncP::load(&cfg, vb.pp("enc_p"))?;
        let ge_to512 = linear(cfg.gin_channels, 512, vb.pp("ge_to512"))?;

        let sv = if is_v2pro {
            Some(ERes2NetV2::load(dir, device, dtype)?)
        } else {
            None
        };
        Ok(Self {
            sv,
            quantizer: SemanticQuantizer::load(vb.clone(), 768, device)?,
            style: StyleEncoder::load(vb, is_v2pro)?,
            enc_p,
            flow,
            dec,
            ge_to512,
            sampling_rate: manifest.audio.sampling_rate,
            hop_length: manifest.audio.hop_length,
            cfg,
            is_v2pro,
            device: device.clone(),
        })
    }

    pub fn sampling_rate(&self) -> usize {
        self.sampling_rate
    }

    pub fn hop_length(&self) -> usize {
        self.hop_length
    }

    pub fn config(&self) -> &SovitsConfig {
        &self.cfg
    }

    /// 参考音频的 HuBERT 特征 `(1, 768, T)` → prompt 语义 codes。
    pub fn prompt_codes(&self, hubert: &Tensor) -> Result<Vec<u32>> {
        self.quantizer.encode(hubert)
    }

    /// 只做 `ssl_proj`（诊断用）
    pub fn project(&self, hubert: &Tensor) -> Result<Tensor> {
        self.quantizer.project(hubert)
    }

    /// codes → 连续语义向量 `(1, 768, T)`。
    pub fn decode_codes(&self, codes: &[u32]) -> Result<Tensor> {
        self.quantizer.decode(codes)
    }

    /// 线性幅度谱 `(1, freq, T)` → 音色向量 `(1, 1024, 1)`。
    pub fn ge(&self, spec: &Tensor, sv: Option<&Tensor>) -> Result<Tensor> {
        self.style.ge(spec, sv)
    }

    pub fn needs_sv(&self) -> bool {
        self.style.needs_sv()
    }

    /// v2Pro：16 kHz 参考音频（**不加 0.3 s 静音尾**，和官方一致）→ `(1, 20480)`
    pub fn sv(&self, audio_16k: &[f32]) -> Result<Tensor> {
        match &self.sv {
            Some(model) => model.compute_sv(audio_16k),
            None => anyhow::bail!("当前模型不需要说话人向量"),
        }
    }

    pub fn is_v2pro(&self) -> bool {
        self.is_v2pro
    }

    /// 诊断：只跑到 `enc_p`，返回 `(m, logs)`（不采样、不过 flow/dec）。
    pub fn probe_enc_p(
        &self,
        codes: &[u32],
        text_ids: &[u32],
        ge: &Tensor,
        speed: f32,
    ) -> Result<(Tensor, Tensor)> {
        let quantized = self.quantizer.decode(codes)?;
        let quantized = if self.cfg.semantic_frame_rate == "25hz" {
            upsample_x2(&quantized)?
        } else {
            quantized
        };
        let ge512 = self
            .ge_to512
            .forward(&ge.transpose(2, 1)?.contiguous()?)?
            .transpose(2, 1)?
            .contiguous()?;
        self.enc_p
            .forward(&quantized, text_ids, &ge512, speed, &self.device)
    }

    /// 诊断：返回 `(z_p, z)`（flow 前 / 后的 latent）
    pub fn probe_flow(
        &self,
        codes: &[u32],
        text_ids: &[u32],
        ge: &Tensor,
        speed: f32,
        noise_scale: f32,
    ) -> Result<(Tensor, Tensor)> {
        let (m, logs) = self.probe_enc_p(codes, text_ids, ge, speed)?;
        let frames = m.dim(2)?;
        let mut rng = Rng(1);
        let mut noise = Vec::with_capacity(frames);
        for _ in 0..frames {
            noise.push(rng.next_normal());
        }
        let noise = Tensor::from_vec(noise, (1, 1, frames), &self.device)?;
        let scale = (logs.exp()? * noise_scale as f64)?;
        let z_p = (&m + noise.broadcast_mul(&scale)?)?;
        let z = self.flow.reverse(&z_p, Some(ge))?;
        Ok((z_p, z))
    }

    /// 诊断：只跑 dec（给什么 latent 就出什么波形）
    pub fn probe_dec(&self, z: &Tensor, ge: &Tensor) -> Result<Tensor> {
        self.dec.forward(z, Some(ge))
    }

    /// AR 出的语义 codes + 音色向量 → 32 kHz 波形 `(1, 1, N)`。
    pub fn synthesize(
        &self,
        codes: &[u32],
        text_ids: &[u32],
        ge: &Tensor,
        opts: &SynthOptions,
    ) -> Result<Tensor> {
        // 1) codes → 连续语义 → ×2 上采样（25 Hz → 50 Hz）
        let quantized = self.quantizer.decode(codes)?;
        let quantized = if self.cfg.semantic_frame_rate == "25hz" {
            upsample_x2(&quantized)?
        } else {
            quantized
        };

        // 2) ge → 512 维（只给 enc_p / MRTE 用）
        let ge512 = self
            .ge_to512
            .forward(&ge.transpose(2, 1)?.contiguous()?)?
            .transpose(2, 1)?
            .contiguous()?;

        // 3) enc_p → (m, logs)
        let (m, logs) = self
            .enc_p
            .forward(&quantized, text_ids, &ge512, opts.speed, &self.device)?;

        // 4) z_p = m + N(0,1) * exp(logs) * noise_scale
        let frames = m.dim(2)?;
        let mut rng = Rng(opts.seed | 1);
        let mut noise = Vec::with_capacity(frames);
        for _ in 0..frames {
            noise.push(rng.next_normal());
        }
        let noise = Tensor::from_vec(noise, (1, 1, frames), &self.device)?;
        let scale = (logs.exp()? * opts.noise_scale as f64)?;
        let z_p = (&m + noise.broadcast_mul(&scale)?)?;

        // 5) flow 逆变换 → 6) HiFiGAN 声码器
        let z = self.flow.reverse(&z_p, Some(ge))?;
        self.dec.forward(&z, Some(ge))
    }
}
