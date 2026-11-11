<p align="center">
  <img src="assets/img/logo.png" alt="aha logo" width="120"/>
</p>

<p align="center">
  <a href="https://github.com/jhqxxx/aha/stargazers">
    <img src="https://img.shields.io/github/stars/jhqxxx/aha" alt="GitHub Stars">
  </a>
  <a href="https://github.com/jhqxxx/aha/issues">
    <img src="https://img.shields.io/github/issues/jhqxxx/aha" alt="GitHub Issues">
  </a>
  <a href="https://github.com/jhqxxx/aha/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/jhqxxx/aha" alt="GitHub License">
  </a>
</p>

<p align="center">
  <a href="README.zh-CN.md">简体中文</a> | <strong>English</strong>
</p>

<p align="center">
  <strong>Official Website:</strong> <a href="https://s5dgj23f.pub.makeit.coderbox.cn/">https://s5dgj23f.pub.makeit.coderbox.cn/</a>
</p>

# aha

**Lightweight AI Inference Engine — All-in-one Solution for Text, Vision, Speech, and OCR**

aha is a high-performance, cross-platform AI inference engine built with Rust and the Candle framework. It brings state-of-the-art AI models to your local machine—no API keys, no cloud dependencies, just pure, fast AI running directly on your hardware.


### Supported Models

| Category | Models |
|----------|--------|
| **Text** | Qwen3, MiniCPM4, MiniCPM5, LFM2, LFM2.5 |
| **Vision** | Qwen2.5-VL, Qwen3-VL, Qwen3.5, <br> LFM2.5-VL, LFM2-VL |
| **OCR** | DeepSeek-OCR, DeepSeek-OCR-2 , PaddleOCR-VL <br> PaddleOCR-VL1.5, PaddleOCR-VL1.6, Hunyuan-OCR <br> GLM-OCR |
| **ASR** | GLM-ASR-Nano, Fun-ASR-Nano, Qwen3-ASR, <br> SenseVoice (sherpa-onnx, `--features sherpa`) |
| **TTS** | VoxCPM, VoxCPM1.5, VoxCPM2, Moss-TTS-Nano, <br> Kokoro (sherpa-onnx, `--features sherpa`) |
| **Image** | RMBG-2.0 (background removal) |
| **Embedding** | Qwen3-Embedding, all-MiniLM-L6-v2 |
| **Reranker** | Qwen3-Reranker |

## Changelog
### 2026-06-06
- add PaddleOCR-VL-1.6

### 2026-05-29
- generate code refactored

### 2026-05-28
- generate code refactoring progress 1/3

### 2026-05-27
- add MiniCPM5

### 2026-05-24
- update doc

### 2026-05-11
- add Moss-TTS-Nano，its performance is worse than the original Python version


**[View full changelog](docs/changelog.md)** →

## Why aha?
- **🚀 High-Performance Inference** - Powered by Candle framework for efficient tensor computation and model inference
- **🔧 Unified Interface** — One tool for text, vision, speech, and OCR
- **📦 Local-First** — All processing runs locally, no data leaves your machine
- **🎯 Cross-Platform** — Works on Linux, macOS, and Windows
- **⚡ GPU Accelerated** — Optional CUDA support for faster inference
- **🛡️ Memory Safe** — Built with Rust for reliability
- **🧠 Attention Optimization** - Optional Flash Attention support for optimized long sequence processing

## Quick Start

### Installation

```bash
git clone https://github.com/jhqxxx/aha.git
cd aha
cargo build --release
```

**Optional Features:**

```bash
# CUDA (NVIDIA GPU acceleration)
cargo build --release --features cuda

# Metal (Apple GPU acceleration for macOS)
cargo build --release --features metal

# Flash Attention (faster inference)
cargo build --release --features cuda,flash-attn

# FFmpeg (multimedia processing)
cargo build --release --features ffmpeg

# sherpa-onnx backends: SenseVoice ASR + Kokoro TTS
# First build downloads a prebuilt C++ static library (~117 MB).
# Set SHERPA_ONNX_ARCHIVE_DIR to reuse an already downloaded archive.
cargo build --release --features sherpa
```

> **sherpa-onnx note**: the `sherpa` feature links the sherpa-onnx **shared** library,
> so the produced binary needs `onnxruntime.dll` and `sherpa-onnx-*.dll` next to it
> (they are copied into `target/<profile>/` by the build script).
> Models for these two backends are **not** hosted on ModelScope — pass a local
> directory with `--weight-path`.

**Android (aarch64)**

Cross-compiling to `aarch64-linux-android` needs an NDK (r26) and `ninja`; the script
resolves the NDK, fixes up CMake (Ninja + CMake's own Android support) and the
`+fp16` flag candle needs:

```bash
pwsh scripts/build-android.ps1 -Release        # -> target/aarch64-linux-android/release/aha
```

The result is arm64-v8a, API 26+, and assumes FP16 arithmetic (ARMv8.2+, i.e. phones
from ~2018 on). Ship `libc++_shared.so` alongside it.

With `-Features sherpa` the build also links `libsherpa-onnx-c-api.so`, so the binary
needs `libsherpa-onnx-c-api.so` + `libonnxruntime.so` next to it on the device. That
crate downloads its prebuilt libs from GitHub at build time; if GitHub is unreachable,
fetch `sherpa-onnx-v<ver>-android.tar.bz2` by hand and set `SHERPA_ONNX_ARCHIVE_DIR`.
Set `HOME` and `AHA_HOME` to a writable directory before running: Android's `home_dir()`
is `/`, so the default `~/.aha` is not writable.

### CLI Quick Reference

```bash

# List all supported models
aha list

# Download model only
aha download -m Qwen/Qwen3-ASR-0.6B

# Download model and start service
aha cli -m Qwen/Qwen3-ASR-0.6B

# Run inference directly (without starting service)
aha run -m Qwen/Qwen3-ASR-0.6B -i "audio.wav"

# Run local all-MiniLM-L6-v2 embedding (native safetensors)
aha run -m all-minilm-l6-v2 -i "Rust embedding test" --weight-path D:\model_download\all-MiniLM-L6-v2

# Start service only (model already downloaded)
aha serv -m Qwen/Qwen3-ASR-0.6B -p 10100
```

### Chat

```bash
aha serv -m Qwen/Qwen3-0.6B -p 10100
```

Then use the unified (OpenAI-compatible) API:

```bash
curl http://localhost:10100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }
'
```

## Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](docs/getting-started.md) | First steps with aha |
| [Installation](docs/installation.md) | Detailed installation guide |
| [CLI Reference](docs/cli.md) | Command-line interface |
| [API Documentation](docs/api.md) | Library & REST API |
| [Supported Models](docs/supported-models.md) | Available AI models |
| [Concepts](docs/concepts.md) | Architecture & design |
| [Development](docs/development.md) | Contributing guide |
| [Changelog](docs/changelog.md) | Version history |


## Development

### Using aha as a Library
> cargo add aha

```rust
// VoxCPM example
use aha::models::voxcpm::generate::VoxCPMGenerate;
use aha::utils::audio_utils::save_wav_mono;
use anyhow::Result;

fn main() -> Result<()> {
    let model_path = "xxx/OpenBMB/VoxCPM2/";

    let mut voxcpm_generate = VoxCPMGenerate::init(model_path, None, None)?;
    let generate = voxcpm_generate.inference(
        "aha是一个基于Rust和Candle框架的本地AI推理引擎，支持多模态模型（文本、视觉、语音、OCR）。".to_string(),
        None,
        None,
        2,
        1000,
        10,
        2.0,
        6.0,
    )?;

    save_wav_mono(&generate, "voxcpm2.wav", voxcpm_generate.sample_rate() as u32)?;
    Ok(())
}
```

### Extending New Models

- Create new model file in src/models/
- Export in src/models/mod.rs
- Add support for CLI model inference in src/exec/
- Add tests and examples in tests/

## Features

- High-performance inference via Candle framework
- Multi-modal model support (vision, language, speech)
- Clean, easy-to-use API design
- Minimal dependencies, compact binaries
- Flash Attention support for long sequences
- FFmpeg support for multimedia processing

## License

Apache-2.0 &mdash; See [LICENSE](LICENSE) for details.

## Acknowledgments

- [Candle](https://github.com/huggingface/candle) - Excellent Rust ML framework
- All model authors and contributors

## Wechat & Donate
<div align="center">

| Wechat Group | Donate |
|--------------|--------|
| ![Wechat Group](./assets/img/aha_weixinqun.png) | ![Donate](./assets/img/donate.png) |

</div>

---

<p align="center">
  <sub>Built with ❤️ by the aha team</sub>
</p>

<p align="center">
  <sub>We're continuously expanding our model support. Contributions are welcome!</sub>
</p>
<p align="center">
  <sub>If this project helps you, please consider giving us a ⭐ Star!</sub>
</p>
