//! IndexTTS 2.5 的音频前端。
//!
//! 三套特征，全部按上游（`infer_v2_5.py` + transformers/librosa/torchaudio）复刻：
//!
//! | 用途 | 采样率 | 规格 |
//! |---|---|---|
//! | w2v-bert 输入 | 16k | Kaldi fbank（400/160/512、povey、preemph 0.97、80 bins、mel floor 1.19e-7），
//! |                |     | 每个 mel bin 沿时间归一化，再 stride=2 拼成 160 维 |
//! | campplus 输入 | 16k | 同样的 Kaldi fbank，之后沿时间维减均值 |
//! | s2mel ref_mel | 22.05k | librosa slaney mel（hann 1024/256、center=False、reflect pad 384、magnitude + log） |

use anyhow::{Result, bail};
use candle_core::{Device, Tensor};

/// 读 wav（单声道 float32，多声道取第一声道）
pub fn load_wav(path: &str) -> Result<(Vec<f32>, usize)> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let sr = spec.sample_rate as usize;
    let channels = spec.channels as usize;
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max))
                .collect::<std::result::Result<Vec<_>, _>>()?
        }
    };
    let mono = if channels <= 1 {
        samples
    } else {
        samples
            .chunks(channels)
            .map(|c| c.iter().sum::<f32>() / channels as f32)
            .collect()
    };
    Ok((mono, sr))
}

pub fn write_wav(path: &str, samples: &[f32], sr: usize) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sr as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec)?;
    for s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        w.write_sample(v)?;
    }
    w.finalize()?;
    Ok(())
}

/// 加窗 sinc 重采样（Hann 窗 16 个过零点，质量足够，且不依赖外部 crate）
pub fn resample(xs: &[f32], from: usize, to: usize) -> Vec<f32> {
    if from == to || xs.is_empty() {
        return xs.to_vec();
    }
    let ratio = to as f64 / from as f64;
    let out_len = (xs.len() as f64 * ratio).round() as usize;
    let cutoff = ratio.min(1.0);
    let taps = 16usize;
    let half = (taps as f64 / cutoff) as isize;
    let mut out = vec![0f32; out_len];
    for (i, o) in out.iter_mut().enumerate() {
        let center = i as f64 / ratio;
        let base = center.floor() as isize;
        let mut acc = 0f64;
        let mut wsum = 0f64;
        for k in -half..=half {
            let idx = base + k;
            if idx < 0 || idx as usize >= xs.len() {
                continue;
            }
            let t = center - idx as f64;
            let sinc = if t.abs() < 1e-9 {
                1.0
            } else {
                let arg = std::f64::consts::PI * t * cutoff;
                arg.sin() / arg
            };
            // Hann 窗，作用在归一化距离上
            let d = (t.abs() / half as f64).min(1.0);
            let w = 0.5 + 0.5 * (std::f64::consts::PI * d).cos();
            acc += ln_val(xs, idx) as f64 * sinc * w;
            wsum += sinc * w;
        }
        *o = if wsum.abs() > 1e-9 {
            (acc / wsum) as f32
        } else {
            0.0
        };
    }
    out
}

fn ln_val(xs: &[f32], idx: isize) -> f32 {
    xs[idx.clamp(0, xs.len() as isize - 1) as usize]
}

// ---------------------------------------------------------------- FFT

/// 原地 radix-2 FFT
fn fft_inplace(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * std::f64::consts::PI / len as f64;
        let (wr, wi) = (ang.cos(), ang.sin());
        let mut i = 0;
        while i < n {
            let mut cur_r = 1.0f64;
            let mut cur_i = 0.0f64;
            for k in 0..len / 2 {
                let ur = re[i + k] as f64;
                let ui = im[i + k] as f64;
                let vr = re[i + k + len / 2] as f64 * cur_r - im[i + k + len / 2] as f64 * cur_i;
                let vi = re[i + k + len / 2] as f64 * cur_i + im[i + k + len / 2] as f64 * cur_r;
                re[i + k] = (ur + vr) as f32;
                im[i + k] = (ui + vi) as f32;
                re[i + k + len / 2] = (ur - vr) as f32;
                im[i + k + len / 2] = (ui - vi) as f32;
                let nr = cur_r * wr - cur_i * wi;
                cur_i = cur_r * wi + cur_i * wr;
                cur_r = nr;
            }
            i += len;
        }
        len <<= 1;
    }
}

fn hann_window(n: usize, periodic: bool) -> Vec<f32> {
    let denom = if periodic { n as f64 } else { (n - 1) as f64 };
    (0..n)
        .map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / denom).cos()) as f32)
        .collect()
}

fn povey_window(n: usize) -> Vec<f32> {
    let denom = (n - 1) as f64;
    (0..n)
        .map(|i| {
            let v = 2.0 * std::f64::consts::PI * i as f64 / denom;
            (0.5 - 0.5 * v.cos()).powf(0.85) as f32
        })
        .collect()
}

// ---------------------------------------------------------------- Kaldi fbank

/// Kaldi mel: 1127 * ln(1 + f/700)
fn hz_to_mel_kaldi(f: f64) -> f64 {
    1127.0 * (1.0 + f / 700.0).ln()
}

fn mel_to_hz_kaldi(m: f64) -> f64 {
    700.0 * (m / 1127.0).exp_m1()
}

/// Kaldi 风格三角滤波器（在 mel 域上做三角），返回 (num_bins, fft/2+1)
fn kaldi_mel_bank(
    sr: usize,
    fft_len: usize,
    num_bins: usize,
    low: f64,
    high: f64,
) -> Vec<Vec<f32>> {
    let n_bins = fft_len / 2 + 1;
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

/// Kaldi fbank 的公共实现（energy/power → mel → log）
fn kaldi_fbank(
    audio: &[f32],
    sr: usize,
    num_bins: usize,
    frame_len: usize,
    hop: usize,
    fft_len: usize,
    log_floor: f64,
) -> Result<Vec<Vec<f32>>> {
    let window = povey_window(frame_len);
    let mel_bank = kaldi_mel_bank(sr, fft_len, num_bins, 20.0, sr as f64 / 2.0);
    let scaled: Vec<f32> = audio.iter().map(|v| v * 32768.0).collect();
    if scaled.len() < frame_len {
        bail!("音频太短（{} < {frame_len}）", scaled.len());
    }
    let frames = (scaled.len() - frame_len) / hop + 1;
    let mut out = Vec::with_capacity(frames);
    let mut re = vec![0f32; fft_len];
    let mut im = vec![0f32; fft_len];
    for f in 0..frames {
        let start = f * hop;
        let mut buf = scaled[start..start + frame_len].to_vec();
        // remove DC offset
        let mean = buf.iter().sum::<f32>() / frame_len as f32;
        for v in buf.iter_mut() {
            *v -= mean;
        }
        // preemphasis
        for i in (1..frame_len).rev() {
            buf[i] -= 0.97 * buf[i - 1];
        }
        buf[0] *= 1.0 - 0.97;
        // window
        for i in 0..frame_len {
            buf[i] *= window[i];
        }
        for i in 0..fft_len {
            re[i] = if i < frame_len { buf[i] } else { 0.0 };
            im[i] = 0.0;
        }
        fft_inplace(&mut re, &mut im);
        let power: Vec<f32> = (0..fft_len / 2 + 1)
            .map(|i| re[i] * re[i] + im[i] * im[i])
            .collect();
        let mut fb = Vec::with_capacity(num_bins);
        for bank in mel_bank.iter() {
            let e: f32 = bank.iter().zip(power.iter()).map(|(w, p)| w * p).sum();
            fb.push((e as f64).max(log_floor).ln() as f32);
        }
        out.push(fb);
    }
    Ok(out)
}

/// w2v-bert 的输入特征：Kaldi fbank + 每个 bin 沿时间归一化 + stride=2 拼接 → (T/2, 160)
pub fn seamless_features(audio_16k: &[f32], device: &Device) -> Result<Tensor> {
    let mut feat = kaldi_fbank(
        audio_16k,
        16000,
        80,
        400,
        160,
        512,
        1.192_092_955_078_125e-7,
    )?;
    let frames = feat.len();
    if frames < 2 {
        bail!("音频太短，fbank 只有 {frames} 帧");
    }
    // 每个 mel bin 沿时间做零均值单位方差
    for b in 0..80 {
        let vals: Vec<f32> = (0..frames).map(|f| feat[f][b]).collect();
        let mean = vals.iter().sum::<f32>() / frames as f32;
        let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / frames as f32;
        let std = (var + 1e-7).sqrt();
        for f in 0..frames {
            feat[f][b] = (feat[f][b] - mean) / std;
        }
    }
    let out_frames = frames / 2;
    let mut data = Vec::with_capacity(out_frames * 160);
    for f in 0..out_frames {
        data.extend_from_slice(&feat[2 * f]);
        data.extend_from_slice(&feat[2 * f + 1]);
    }
    Ok(Tensor::from_vec(data, (1, out_frames, 160), device)?)
}

/// campplus 的输入：Kaldi fbank 后沿时间维减均值 → (T, 80)
pub fn campplus_fbank(audio_16k: &[f32], device: &Device) -> Result<Tensor> {
    let mut feat = kaldi_fbank(audio_16k, 16000, 80, 400, 160, 512, 1e-10)?;
    let frames = feat.len();
    for b in 0..80 {
        let mean = (0..frames).map(|f| feat[f][b]).sum::<f32>() / frames as f32;
        for f in 0..frames {
            feat[f][b] -= mean;
        }
    }
    let mut data = Vec::with_capacity(frames * 80);
    for row in feat {
        data.extend_from_slice(&row);
    }
    Ok(Tensor::from_vec(data, (1, frames, 80), device)?)
}

// ---------------------------------------------------------------- librosa slaney mel（s2mel）

fn hz_to_mel_slaney(f: f64) -> f64 {
    let f_min = 0.0;
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep = 6.4f64.ln() / 27.0;
    if f >= min_log_hz {
        min_log_mel + (f / min_log_hz).ln() / logstep
    } else {
        (f - f_min) / f_sp
    }
}

fn mel_to_hz_slaney(m: f64) -> f64 {
    let f_min = 0.0;
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep = 6.4f64.ln() / 27.0;
    if m >= min_log_mel {
        min_log_hz * (logstep * (m - min_log_mel)).exp()
    } else {
        f_min + f_sp * m
    }
}

/// librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax)（htk=False, norm="slaney"）
fn slaney_mel_bank(sr: usize, n_fft: usize, n_mels: usize, fmin: f64, fmax: f64) -> Vec<Vec<f32>> {
    let n_bins = n_fft / 2 + 1;
    let mel_min = hz_to_mel_slaney(fmin);
    let mel_max = hz_to_mel_slaney(fmax);
    let freqs: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz_slaney(mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64))
        .collect();
    let fft_freqs: Vec<f64> = (0..n_bins)
        .map(|i| i as f64 * sr as f64 / n_fft as f64)
        .collect();
    (0..n_mels)
        .map(|m| {
            let (lower, center, upper) = (freqs[m], freqs[m + 1], freqs[m + 2]);
            let enorm = 2.0 / (upper - lower);
            fft_freqs
                .iter()
                .map(|f| {
                    let v = if *f < lower || *f > upper {
                        0.0
                    } else if *f <= center {
                        (*f - lower) / (center - lower)
                    } else {
                        (upper - *f) / (upper - center)
                    };
                    (v * enorm) as f32
                })
                .collect()
        })
        .collect()
}

/// s2mel 的 ref_mel：22.05k，n_fft/win 1024、hop 256、80 mel、center=False、log(clamp 1e-5)
pub fn mel_spectrogram_22050(audio: &[f32], device: &Device) -> Result<Tensor> {
    let (n_fft, win, hop, n_mels) = (1024usize, 1024usize, 256usize, 80usize);
    let pad = (n_fft - hop) / 2; // 384
    let mut padded = Vec::with_capacity(audio.len() + 2 * pad);
    // reflect pad
    for i in (1..=pad).rev() {
        padded.push(audio[i.min(audio.len() - 1)]);
    }
    padded.extend_from_slice(audio);
    for i in (1..=pad).rev() {
        padded.push(audio[audio.len().saturating_sub(1 + i)]);
    }
    let window = hann_window(win, true);
    let mel_bank = slaney_mel_bank(22050, n_fft, n_mels, 0.0, 22050.0 / 2.0);
    if padded.len() < n_fft {
        bail!("音频太短，无法做 1024 点 STFT");
    }
    let frames = (padded.len() - n_fft) / hop + 1;
    let mut mel = vec![0f32; n_mels * frames];
    let mut re = vec![0f32; n_fft];
    let mut im = vec![0f32; n_fft];
    for f in 0..frames {
        let start = f * hop;
        for i in 0..n_fft {
            re[i] = padded[start + i] * window[i];
            im[i] = 0.0;
        }
        fft_inplace(&mut re, &mut im);
        let mag: Vec<f32> = (0..n_fft / 2 + 1)
            .map(|i| (re[i] * re[i] + im[i] * im[i] + 1e-9).sqrt())
            .collect();
        for m in 0..n_mels {
            let e: f32 = mel_bank[m].iter().zip(mag.iter()).map(|(w, v)| w * v).sum();
            mel[m * frames + f] = (e as f64).max(1e-5).ln() as f32;
        }
    }
    Ok(Tensor::from_vec(mel, (1, n_mels, frames), device)?)
}
