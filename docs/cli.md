# CLI Reference

Complete command-line interface reference for aha.

AHA is a high-performance model inference library based on the Candle framework, supporting various multimodal models including vision, language, and audio models.

```bash
aha [COMMAND] [OPTIONS]
```

## Global Options

| Option | Description | Default |
|--------|-------------|---------|
| `-a, --address <ADDRESS>` | Service listen address | 127.0.0.1 |
| `-p, --port <PORT>` | Service listen port | 10100 |
| `-m, --model <MODEL>` | Model type (required) | - |
| `--weight-path <WEIGHT_PATH>` | Local model weight path | - |
| `--save-dir <SAVE_DIR>` | Model download save directory | ~/.aha/ |
| `--download-retries <DOWNLOAD_RETRIES>` | Download retry count | 3 |
| `--gguf-path <GGUF_PATH>` | Local GGUF weight（required when using GGUF models） | - |
| `--mmproj-path <MMPROJ_PATH>` | Local mmproj GGUF weight | - |
| `--onnx-path <ONNX_PATH>` | Local ONNX weight（required when using ONNX models） | - |
| `--config-path <ONNX_PATH>` | extra config path for gguf/onnx | - |
| `--no-warmup` | Disable startup warmup (on by default). Warmup replays each role once before the port opens, so the first real request isn't slower than the rest | off |
| `--provider <PROVIDER>` | ONNX Runtime execution provider for SenseVoice / Kokoro: `cpu`, `qnn`, `nnapi`, `cuda`, `coreml`, `trt`, `rknn`, `openvino`, `npu` to pick whatever the probe recommends, or `plugin[:<conf>]` to load a runtime plugin EP such as Qualcomm`s QNN plugin. Silently falls back to CPU when the runtime lacks it. Falls back to `AHA_PROVIDER` when omitted. See [NPU Acceleration](./npu.md) | auto |
| `-h, --help` | Display help information | - |
| `-V, --version` | Display version number | - |

## Commands

### cli - Download model and start service 

Download the specified model and start an HTTP service. 
Download the specified model (from ModelScope when no `--weight-path` is given) and start an HTTP service.
`qwen3.5-gguf` defaults to the 0.8B Q4_K_M mirror (you can still pass `--gguf-path`); ONNX models must be local paths.

**Syntax:**
```bash
aha cli [OPTIONS] --model <MODEL>
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `-a, --address <ADDRESS>` | Service listen address | 127.0.0.1 |
| `-p, --port <PORT>` | Service listen port | 10100 |
| `-m, --model <MODEL>` | Model type (required) | - |
| `--weight-path <WEIGHT_PATH>` | Local model weight path (skip download if specified) | - |
| `--save-dir <SAVE_DIR>` | Model download save directory | ~/.aha/ |
| `--download-retries <DOWNLOAD_RETRIES>` | Download retry count | 3 |
| `--gguf-path <GGUF_PATH>` | Local GGUF weight（required when using GGUF models） | - |
| `--mmproj-path <MMPROJ_PATH>` | Local mmproj GGUF weight | - |
| `--onnx-path <ONNX_PATH>` | Local ONNX weight（required when using ONNX models） | - |
| `--config-path <ONNX_PATH>` | extra config path for gguf/onnx | - |

**Examples:**

```bash
# Download model and start service (default port 10100)
aha cli -m Qwen/Qwen3-VL-2B-Instruct

# Specify port and save directory
aha cli -m Qwen/Qwen3-VL-2B-Instruct -p 8080 --save-dir /data/models

# Use local model (skip download)
aha cli -m Qwen/Qwen3-VL-2B-Instruct --weight-path /path/to/model

# use gguf-path and mmproj-path
aha cli -m qwen3.5-gguf --gguf-path /path/to/xxx.gguf --mmproj-path /path/to/mmproj-xxx.gguf

```

### run - Direct model inference

Run model inference directly without starting an HTTP service. Suitable for one-time inference tasks or batch processing.

**Syntax:**
```bash
aha run [OPTIONS] --model <MODEL> --input <INPUT> [--input <INPUT2>] [--weight-path <WEIGHT_PATH>] [--gguf-path <GGUF_PATH>] [--mmproj-path <MMPROJ_PATH>] [--onnx-path <ONNX_PATH>] [--config-path <CONFIG_PATH>]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model <MODEL>` | Model type (required) | - |
| `-i, --input <INPUT>` | Input text or file path (model-specific interpretation, supports 1-2 parameters: input1: prompt text, input2: file path) | - |
| `-o, --output <OUTPUT>` | Output file path (optional, auto-generated if not specified) | - |
| `--weight-path <WEIGHT_PATH>` | Local model weight path (required when using safetensors models) | - |
| `--gguf-path <GGUF_PATH>` | Local GGUF model weight path（required when using GGUF models） | - |
| `--mmproj-path <MMPROJ_PATH>` | Local mmproj GGUF weight path（optional，If not specified, the module will not be loaded） | - |
| `--onnx-path <ONNX_PATH>` | Local ONNX weight（required when using ONNX models） | - |
| `--config-path <ONNX_PATH>` | extra config path for gguf/onnx | - |

**Examples:**

```bash
# VoxCPM1.5 text-to-speech (single input)
aha run -m OpenBMB/VoxCPM1.5 -i "太阳当空照" -o output.wav --weight-path /path/to/model

# VoxCPM1.5 read input from file (single input)
aha run -m OpenBMB/VoxCPM1.5 -i "file://./input.txt" --weight-path /path/to/model

# MiniCPM4 text generation (single input)
aha run -m OpenBMB/MiniCPM4-0.5B -i "你好" --weight-path /path/to/model

# DeepSeek OCR image recognition (single input)
aha run -m deepseek-ai/DeepSeek-OCR -i "image.jpg" --weight-path /path/to/model

# RMBG2.0 background removal (single input)
aha run -m AI-ModelScope/RMBG-2.0 -i "photo.png" -o "no_bg.png" --weight-path /path/to/model

# GLM-ASR speech recognition (two inputs: prompt text + audio file)
aha run -m ZhipuAI/GLM-ASR-Nano-2512 -i "请转写这段音频" -i "audio.wav" --weight-path /path/to/model

# Fun-ASR speech recognition (two inputs: prompt text + audio file)
aha run -m FunAudioLLM/Fun-ASR-Nano-2512 -i "语音转写：" -i "audio.wav" --weight-path /path/to/model

# qwen3 text generation (single input)
aha run -m Qwen/Qwen3-0.6B -i "你好" --weight-path /path/to/model

# qwen2.5vl image understanding (two inputs: prompt text + image file)
aha run -m Qwen/Qwen2.5-VL-3B-Instruct -i "请分析图片并提取所有可见文本内容，按从左到右、从上到下的布局，返回纯文本" -i "image.jpg" --weight-path /path/to/model

# Qwen3-ASR speech recognition (single input: audio file)
aha run -m Qwen/Qwen3-ASR-0.6B -i "audio.wav" --weight-path /path/to/model

# Qwen3.5-GGUF without mmproj (single input: prompt text)
aha run -m qwen3.5-gguf -i 你如何看待AI --gguf-path /path/to/xxx.gguf

# Qwen3.5-GGUF with mmproj (two inputs：prompt text + file)
aha run -m qwen3.5-gguf -i 提取图片中的文本 -i https://ai.bdstatic.com/file/C56CC9B274CF460CA33
63E59ECD94423 --gguf-path /path/to/xxx.gguf --mmproj-path /path/to/mmproj-xxx.gguf


```

### serv - Start service

Start HTTP service with a model. 
Safetensors model: The `--weight-path` is optional - if not specified, it defaults to `~/.aha/{model_id}`
and is downloaded from ModelScope on first use (pet/local-converted models use their own mirrors).
`qwen3.5-gguf` without `--gguf-path` downloads the 0.8B Q4_K_M mirror; ONNX models must be local paths.

**Syntax:**
```bash
aha serv [OPTIONS] --model <MODEL> [--weight-path <WEIGHT_PATH>] [--gguf-path <GGUF_PATH>] [--mmproj-path <MMPROJ_PATH>] [--onnx-path <ONNX_PATH>] [--config-path <CONFIG_PATH>]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `-a, --address <ADDRESS>` | Service listen address | 127.0.0.1 |
| `-p, --port <PORT>` | Service listen port | 10100 |
| `-m, --model <MODEL>` | Model type (required) | - |
| `--weight-path <WEIGHT_PATH>` | Local model weight path (optional) | ~/.aha/{model_id} |
| `--allow-remote-shutdown` | Allow remote shutdown requests (not recommended) | false |
| `--gguf-path <GGUF_PATH>` | Local GGUF model weight path（required when using GGUF models） | - |
| `--mmproj-path <MMPROJ_PATH>` | Local mmproj GGUF weight path（optional，If not specified, the module will not be loaded） | - |
| `--onnx-path <ONNX_PATH>` | Local ONNX model directory/file path（required when using ONNX models） | - |
| `--config-path <ONNX_PATH>` | extra config path for gguf/onnx | - |

**Examples:**

```bash
# Start service with default model path (~/.aha/{model_id})
aha serv -m Qwen/Qwen3-VL-2B-Instruct

# Start service with local model
aha serv -m Qwen/Qwen3-VL-2B-Instruct --weight-path /path/to/model

# Start with specified port
aha serv -m Qwen/Qwen3-VL-2B-Instruct -p 8080

# Specify listen address
aha serv -m Qwen/Qwen3-VL-2B-Instruct -a 0.0.0.0

# Enable remote shutdown (not recommended for production)
aha serv -m Qwen/Qwen3-VL-2B-Instruct --allow-remote-shutdown
```

### ps - List running services

List all currently running AHA services with their process IDs, ports, and status.

**Syntax:**
```bash
aha ps [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `-c, --compact` | Compact output format (show service IDs only) | false |

**Examples:**

```bash
# List all running services (table format)
aha ps

# Compact output (service IDs only)
aha ps -c
```

**Output Format:**

```
Service ID           PID        Model                Port       Address         Status
-------------------------------------------------------------------------------------
56860@10100          56860      N/A                  10100      127.0.0.1       Running
```

**Fields:**
- `Service ID`: Unique identifier in format `pid@port`
- `PID`: Process ID
- `Model`: Model name (N/A if not detected)
- `Port`: Service port number
- `Address`: Service listen address
- `Status`: Service status (Running, Stopping, Unknown)

### download - Download model

Download the specified model only, without starting the service.

**Syntax:**
```bash
aha download [OPTIONS] --model <MODEL>
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model <MODEL>` | Model type (required) | - |
| `-s, --save-dir <SAVE_DIR>` | Model download save directory | ~/.aha/ |
| `--download-retries <DOWNLOAD_RETRIES>` | Download retry count | 3 |

**Examples:**

```bash
# Download model to default directory
aha download -m Qwen/Qwen3-VL-2B-Instruct

# Specify save directory
aha download -m Qwen/Qwen3-VL-2B-Instruct -s /data/models

# Specify download retry count
aha download -m Qwen/Qwen3-VL-2B-Instruct --download-retries 5

# Download MiniCPM4-0.5B model
aha download -m OpenBMB/MiniCPM4-0.5B -s models
```

### delete - Delete downloaded model

Delete a downloaded model from the default location (`~/.aha/{model_id}`).

**Syntax:**
```bash
aha delete [OPTIONS] --model <MODEL>
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model <MODEL>` | Model type (required) | - |

**Examples:**

```bash
# Delete RMBG2.0 model from default location
aha delete -m AI-ModelScope/RMBG-2.0

# Delete Qwen3-VL-2B model
aha delete --model Qwen/Qwen3-VL-2B-Instruct
```

**Behavior:**
- Displays model information (ID, location, size) before deletion
- Requires confirmation (y/N) before proceeding
- Shows "Model not found" message if the model directory doesn't exist
- Shows "Model deleted successfully" message after completion

### list - List all supported models

List all supported models with their ModelScope IDs.

**Syntax:**
```bash
aha list [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `-j, --json` | Output in JSON format (includes name, model_id, and type fields) | false |

**Examples:**

```bash
# List models in table format (default)
aha list

# List models in JSON format
aha list --json

# Short form
aha list -j
```

**JSON Output Format:**

When using `--json`, the output includes:
- `name`: Model identifier used with `-m` flag
- `model_id`: Full ModelScope model ID
- `type`: Model category (`llm`, `ocr`, `asr`, or `image`)

Example:
```json
[
  {
    "name": "Qwen/Qwen3-VL-2B-Instruct",
    "model_id": "Qwen/Qwen3-VL-2B-Instruct",
    "type": "llm"
  },
  {
    "name": "deepseek-ai/DeepSeek-OCR",
    "model_id": "deepseek-ai/DeepSeek-OCR",
    "type": "ocr"
  }
]
```

**Model Types:**
- `llm`: Language models (text generation, chat, etc.)
- `ocr`: Optical Character Recognition models
- `asr`: Automatic Speech Recognition models
- `image`: Image processing models
- `tts`： Text to speech

### npu - Inspect the NPU runtime on this machine

Report what this machine offers an ONNX Runtime provider: Qualcomm QNN (Hexagon HTP), NNAPI, and which execution providers the ONNX Runtime next to the binary was built with. It also says whether this particular `aha` build can use a provider at all.

**Syntax:**
```bash
aha npu [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `-j, --json` | Output in JSON format | false |
| `-r, --recommended` | Print only the recommended `--provider` value (`cpu` when nothing is usable) | false |

**Examples:**

```bash
# Full report
aha npu

# Just the value to pass to --provider
aha npu -r

# Point the probe at a custom ONNX Runtime build
AHA_NPU_LIB_DIR=/data/local/tmp/qnn-ort aha npu --json
```

**Output:** the report lists each runtime it found with its path, the HTP architecture parsed from `libQnnHtpV{N}Stub.so`, the execution providers embedded in the local ONNX Runtime, and notes explaining what would go wrong (missing `libQnnSystem.so`, an ONNX Runtime without the QNN EP, a dead NNAPI service).

Only SenseVoice and Kokoro run through ONNX Runtime, so they are the only models `--provider` can accelerate. See [NPU Acceleration](./npu.md) for the full picture and for what has to be built by hand.

### qnn - Inspect / run a QAIRT model

Loads a `.bin` (produced by `scripts/qnn_convert.py --stage`, or any QAIRT context binary) or a model library `lib<name>.so` (from `qnn_convert.py --build-so`, with the weights in `<name>.bin` next to it), prints the graphs and tensors inside, and optionally runs one inference. It knows **no model specifics**: graph names, tensor names, types and shapes all come from the binary, so the same command verifies whatever you converted.

**Syntax:**
```bash
aha qnn --context <BIN> [OPTIONS]
aha qnn --model <libNAME.so> [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--context <BIN>` | Path to the context binary (mutually exclusive with `--model`) | — |
| `--model <SO>` | Path to a model library (mutually exclusive with `--context`); weights live in `<name>.bin` next to it | — |
| `--backend <LIB>` | HTP backend library | `libQnnHtp.so` |
| `--system <LIB>` | QNN system library (metadata, `--context` only) | `libQnnSystem.so` |
| `--info` | Print graphs/tensors, do not execute | false |
| `--graph <N>` | Which graph to run | 0 |
| `--input <NAME=PATH>` | Raw bytes for a named tensor, repeatable; zero-filled when omitted | — |
| `--out-dir <DIR>` | Write each output tensor as `<name>.bin` | — |

**Examples:**

```bash
# What is in this binary?
aha qnn --context vitsS_ctx.bin --info

# One real inference, outputs on disk
aha qnn --context vitsS_ctx.bin \
  --input codes=codes.raw --input text=text.raw \
  --input refer=refer.raw --input sv_emb=sv_emb.raw \
  --out-dir out

# Graphs with float32 IO (say GPT-SoVITS' SoVITS decoder) go through a model library
aha qnn --model libvitsF.so --info
```

**Automatic fallback when in-process execution is refused**: on HTP v81 some graphs (SoVITS, whose IO is float32, is one) are rejected by the DSP with `Qnn_Common_Error_General (1100)`. In that case `aha` writes the same inputs as raw files plus an `input_list.txt`, runs the SDK's own `qnn-net-run` in the artifact directory (`--model` or `--retrieve_context`), and reads `Result_0/<tensor>.raw` back into memory — so the command still succeeds (the fallback output is md5-identical to what `qnn-net-run` produces directly). This needs `qnn-net-run` next to the artifact (`scripts/qnn_convert.py --stage` ships it); `AHA_QNN_NETRUN=<path>` overrides where it looks, and `AHA_QNN_NETRUN=0` turns the fallback off so you can tell whether a failure belongs to the shim or the QNN runtime.

After one refusal the same process goes straight to the fallback on later calls, and `AHA_QNN_SKIP_INPROCESS=1` skips even the first attempt (saves ~0.6 s).

**Context binary or model library**: both work, but the model-library path **composes and finalises the graph in every process** — measured at **~737 s** for SoVITS — while the same graph's context binary loads in ~0.7 s and the whole chain takes **~1.8 s**. Keep the model library for conversion/inspection (`--info`, feeding inputs to check shapes) and run business traffic from a context binary (build it once with `qnn-context-binary-generator`).

**Build-time switch:** this subcommand is behind the `qnn` cargo feature (an Android-only backend). `scripts/build_android.py` adds it automatically; desktop builds leave it **off**, so `aha qnn` does not exist there (`aha npu` probing still does). Pass `--features qnn` explicitly to try it on a Linux desktop.

**Requirements:** `libaha_qnn.so` next to the `aha` executable (`scripts/build_android.py` builds and places it whenever `QNN_SDK_ROOT` is set), plus `libQnnHtp.so` / `libQnnSystem.so` on the device. Without the shim the subcommand says exactly what is missing; nothing else is affected.

## Common Use Cases

### Scenario 1: Quick start inference service

```bash
# One command to download and start service
aha cli -m Qwen/Qwen3-VL-2B-Instruct
```

### Scenario 2: Start service with existing model

```bash
# Assuming model is downloaded to /data/models/Qwen/Qwen3-VL-2B-Instruct
aha serv -m Qwen/Qwen3-VL-2B-Instruct --weight-path /data/models/Qwen/Qwen3-VL-2B-Instruct
```

### Scenario 3: Pre-download model

```bash
# Download model to specified directory for later use
aha download -m Qwen/Qwen3-VL-2B-Instruct -s /data/models

# Later start with local model
aha serv -m Qwen/Qwen3-VL-2B-Instruct --weight-path /data/models/Qwen/Qwen3-VL-2B-Instruct
```

### Scenario 4: Custom service port and address

```bash
# Start service on 0.0.0.0:8080, allow external access
aha cli -m Qwen/Qwen3-VL-2B-Instruct -a 0.0.0.0 -p 8080
```

## API Endpoints

After the service starts, the following API endpoints are available:

### Chat Completion Endpoint
- **Endpoint**: `POST /chat/completions`
- **Function**: Multimodal chat and text generation
- **Supported Models**: Qwen2.5VL, Qwen3, Qwen3VL, DeepSeekOCR, GLM-ASR-Nano-2512, Fun-ASR-Nano-2512, etc.
- **Format**: OpenAI Chat Completion format
- **Streaming Support**: Yes

### Image Processing Endpoint
- **Endpoint**: `POST /images/remove_background`
- **Function**: Image background removal
- **Supported Models**: RMBG-2.0
- **Format**: OpenAI Chat Completion format
- **Streaming Support**: No

### Audio Generation Endpoint
- **Endpoint**: `POST /audio/speech`
- **Function**: Speech synthesis and generation
- **Supported Models**: VoxCPM, VoxCPM1.5
- **Format**: OpenAI Chat Completion format
- **Streaming Support**: No

### Embeddings Endpoint
- **Endpoint**: `POST /embeddings` or `POST /v1/embeddings`
- **Function**: Text embedding generation
- **Supported Models**: Qwen3-Embedding family
- **Format**: OpenAI embeddings format
- **Streaming Support**: No

### Rerank Endpoint
- **Endpoint**: `POST /rerank` or `POST /v1/rerank`
- **Function**: Query-document reranking
- **Supported Models**: Qwen3-Reranker family
- **Format**: Rerank JSON response (`results[index,relevance_score,document]`)
- **Streaming Support**: No

### Shutdown Endpoint
- **Endpoint**: `POST /shutdown`
- **Function**: Gracefully shut down the server
- **Security**: Localhost only by default, use `--allow-remote-shutdown` flag to enable remote access (not recommended)
- **Format**: JSON response


## Getting Help

```bash
# View main help
aha --help

# View subcommand help
aha cli --help
aha serv --help
aha download --help

# View version information
aha --version
```

## See Also

- [Getting Started](./getting-started.md) - Quick start guide
- [NPU Acceleration](./npu.md) - `--provider`, `aha npu`, and what an NPU can actually run
- [API Documentation](./api.md) - REST API reference
- [Supported Models](./supported-tools.md) - Available models
