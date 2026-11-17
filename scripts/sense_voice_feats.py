"""SenseVoice 的特征前端：fbank(80) -> LFR(7,6) -> CMVN。

与 sherpa-onnx 的 ``InitFeatConfig`` / ``ApplyLfr`` / ``ApplyCMVN`` 对齐
（``hamming`` 窗、``snip_edges=true``、frame 400 / shift 160、fft 512、
80 mel bins、low 20 / high 8000、preemph 0.97、kaldi mel 公式）。

只依赖 numpy + scipy，用来生成校准数据，不参与推理。
"""

from __future__ import annotations

import wave

import numpy as np
from scipy.signal import resample_poly


def load_audio(path: str, sample_rate: int = 16000) -> np.ndarray:
    """读 wav（16/32 bit，单/多声道），混成单声道并重采样到 16k。"""
    with wave.open(path, "rb") as w:
        frames = w.readframes(w.getnframes())
        channels = w.getnchannels()
        rate = w.getframerate()
        width = w.getsampwidth()

    x = np.frombuffer(frames, dtype={2: "<i2", 4: "<i4"}[width]).astype(np.float32)
    x /= 32768.0 if width == 2 else 2147483648.0
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    if rate != sample_rate:
        x = resample_poly(x, sample_rate, rate).astype(np.float32)
    return x


def _window(name: str, n: int) -> np.ndarray:
    i = np.arange(n)
    if name == "hamming":
        return (0.54 - 0.46 * np.cos(2 * np.pi * i / (n - 1))).astype(np.float32)
    return np.power(0.5 - 0.5 * np.cos(2 * np.pi * i / (n - 1)), 0.85).astype(np.float32)


def _mel(f: np.ndarray | float) -> np.ndarray | float:
    return 1127.0 * np.log(1.0 + f / 700.0)


def _mel_banks(
    num_bins: int = 80,
    fft_size: int = 512,
    sample_rate: int = 16000,
    low: float = 20.0,
    high: float = 8000.0,
) -> np.ndarray:
    bins = fft_size // 2 + 1
    width = sample_rate / fft_size
    lo, hi = _mel(low), _mel(high)
    delta = (hi - lo) / (num_bins + 1)
    banks = np.zeros((num_bins, bins), np.float32)
    for j in range(num_bins):
        left, center, right = lo + j * delta, lo + (j + 1) * delta, lo + (j + 2) * delta
        for k in range(bins):
            m = _mel(k * width)
            if left < m < right:
                banks[j, k] = (m - left) / (center - left) if m <= center else (right - m) / (right - center)
    return banks


def fbank(
    x: np.ndarray,
    frame_length: int = 400,
    frame_shift: int = 160,
    sample_rate: int = 16000,
    num_bins: int = 80,
    fft_size: int = 512,
    preemph: float = 0.97,
    window: str = "hamming",
    snip_edges: bool = True,
) -> np.ndarray:
    win = _window(window, frame_length)
    banks = _mel_banks(num_bins, fft_size, sample_rate)
    n = len(x)
    if snip_edges:
        num_frames = 0 if n < frame_length else 1 + (n - frame_length) // frame_shift
    else:
        num_frames = (n + frame_shift // 2) // frame_shift

    out = np.empty((max(num_frames, 0), num_bins), np.float32)
    for i in range(num_frames):
        start = i * frame_shift if snip_edges else i * frame_shift - frame_length // 2
        a, b = max(0, start), min(n, start + frame_length)
        frame = np.zeros(frame_length, np.float32)
        if b > a:
            frame[a - start : b - start] = x[a:b]
        frame -= frame.mean()
        padded = np.empty_like(frame)
        padded[0] = frame[0] * (1 - preemph)
        padded[1:] = frame[1:] - preemph * frame[:-1]
        spectrum = np.fft.rfft(padded * win, fft_size)
        energy = banks @ (spectrum.real**2 + spectrum.imag**2).astype(np.float32)
        out[i] = np.log(np.maximum(energy, 1e-10))
    return out


def lfr(feat: np.ndarray, window_size: int = 7, window_shift: int = 6) -> np.ndarray:
    """与 sherpa-onnx ``ApplyLfr`` 相同：左侧重复首帧、右侧 clamp 到末帧。"""
    frames, dim = feat.shape
    out_frames = 1 + (frames - 1) // window_shift
    left = (window_size - 1) // 2
    out = np.empty((out_frames, dim * window_size), np.float32)
    for i in range(out_frames):
        center = i * window_shift
        left_pad = left - center if center < left else 0
        first = 0 if center < left else center - left
        row = []
        for j in range(window_size):
            idx = 0
            if j >= left_pad:
                offset = j - left_pad
                idx = frames - 1 if first + offset > frames - 1 else first + offset
            row.append(feat[idx])
        out[i] = np.concatenate(row)
    return out


def feats16k(
    x: np.ndarray,
    neg_mean: np.ndarray,
    inv_stddev: np.ndarray,
    window_size: int = 7,
    window_shift: int = 6,
) -> np.ndarray:
    """wav 采样 -> 模型输入特征 ``[T, 560]``。"""
    feat = fbank(x, window="hamming", snip_edges=True)
    return (lfr(feat, window_size, window_shift) + neg_mean) * inv_stddev
