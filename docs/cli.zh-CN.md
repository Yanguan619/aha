# CLI 参考

aha 的完整命令行界面参考。

AHA 是一个基于 Candle 框架的高性能模型推理库，支持多种多模态模型，包括视觉、语言和语音模型。

```bash
aha [COMMAND] [OPTIONS]
```

## 全局选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-a, --address <ADDRESS>` | 服务监听地址 | 127.0.0.1 |
| `-p, --port <PORT>` | 服务监听端口 | 10100 |
| `-m, --model <MODEL>` | 模型类型（必选） | - |
| `--weight-path <WEIGHT_PATH>` | 本地safetensors模型权重路径 | - |
| `--save-dir <SAVE_DIR>` | 模型下载保存目录 | ~/.aha/ |
| `--download-retries <DOWNLOAD_RETRIES>` | 下载重试次数 | 3 |
| `--gguf-path <GGUF_PATH>` | 本地 GGUF 模型权重（使用GGUF模型时必选） | - |
| `--mmproj-path <MMPROJ_PATH>` | 本地 mmproj GGUF 模型权重（可选，未指定则不加载该模块） | - |
| `--onnx-path <ONNX_PATH>` | 本地 ONNX 模型权重 （使用ONNX模型时必选） | - |
| `--config-path <ONNX_PATH>` | GGUF/ONNX 需要的额外配置路径(可选) | - |
| `--no-warmup` | 关掉启动预热（默认开）。预热会在端口开之前空跑一遍每个角色，把首次推理的一次性开销提前 | 关 |
| `--provider <PROVIDER>` | SenseVoice / Kokoro 用的 ONNX Runtime execution provider：`cpu`、`qnn`、`nnapi`、`cuda`、`coreml`、`trt`、`rknn`、`openvino`，`npu`（用探测建议的那个），或 `plugin[:<conf>]`（加载运行时 plugin EP，比如高通 QNN plugin）。运行时没编这个 EP 时会静默退回 CPU。不传则读环境变量 `AHA_PROVIDER`。详见 [NPU 加速](./npu.zh-CN.md) | auto |
| `-h, --help` | 显示帮助信息 | - |
| `-V, --version` | 显示版本号 | - |

## 子命令

### cli - 下载模型并启动服务

下载指定的模型并启动 HTTP 服务。
没给 `--weight-path` 时会从 ModelScope 自动下载；`qwen3.5-gguf` 默认下 0.8B Q4_K_M
（也可以继续用 `--gguf-path` 指定本地文件），ONNX 模型仍必须指定本地路径。

**语法：**
```bash
aha cli [OPTIONS] --model <MODEL>
```

**选项：**

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-a, --address <ADDRESS>` | 服务监听地址 | 127.0.0.1 |
| `-p, --port <PORT>` | 服务监听端口 | 10100 |
| `-m, --model <MODEL>` | 模型类型（必选） | - |
| `--weight-path <WEIGHT_PATH>` | 本地safetensors模型权重路径（如指定则跳过下载） | - |
| `--save-dir <SAVE_DIR>` | 模型下载保存目录 | ~/.aha/ |
| `--download-retries <DOWNLOAD_RETRIES>` | 下载重试次数 | 3 |
| `--gguf-path <GGUF_PATH>` | 本地 GGUF 模型权重（使用GGUF模型时必选） | - |
| `--mmproj-path <MMPROJ_PATH>` | 本地 mmproj GGUF 模型权重（可选，未指定则不加载该模块） | - |
| `--onnx-path <ONNX_PATH>` | 本地 ONNX 模型权重 （使用ONNX模型时必选） | - |
| `--config-path <ONNX_PATH>` | GGUF/ONNX 需要的额外配置路径(可选) | - |

**示例：**

```bash
# 下载模型并启动服务（默认端口 10100）
aha cli -m Qwen/Qwen3-VL-2B-Instruct

# 指定端口和保存目录
aha cli -m Qwen/Qwen3-VL-2B-Instruct -p 8080 --save-dir /data/models

# 使用本地模型（不下载）
aha cli -m Qwen/Qwen3-VL-2B-Instruct --weight-path /path/to/model

# 指定gguf-path和mmproj-path
aha cli -m qwen3.5-gguf --gguf-path /path/to/xxx.gguf --mmproj-path /path/to/mmproj-xxx.gguf

```

### run - 直接模型推理

直接运行模型推理，无需启动 HTTP 服务。适用于一次性推理任务或批处理。

**语法：**
```bash
aha run [OPTIONS] --model <MODEL> --input <INPUT> [--input <INPUT2>] [--weight-path <WEIGHT_PATH>] [--gguf-path <GGUF_PATH>] [--mmproj-path <MMPROJ_PATH>] [--onnx-path <ONNX_PATH>] [--config-path <CONFIG_PATH>]
```

**选项：**

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-m, --model <MODEL>` | 模型类型（必选） | - |
| `-i, --input <INPUT>` | 输入文本或文件路径（模型特定解释，支持1-2个参数, input1： 提示文本, input2: 文件地址） | - |
| `-o, --output <OUTPUT>` | 输出文件路径（可选，未指定则自动生成） | - |
| `--weight-path <WEIGHT_PATH>` | 本地模型权重路径（使用Safetensors模型时必选） | - |
| `--gguf-path <GGUF_PATH>` | 本地GGUF模型权重路径（使用GGUF模型时必选） | - |
| `--mmproj-path <MMPROJ_PATH>` | 本地mmproj GGUF模型权重路径（可选，未指定则不加载该模块） | - |
| `--onnx-path <ONNX_PATH>` | 本地 ONNX 模型权重 （使用ONNX模型时必选） | - |
| `--config-path <ONNX_PATH>` | GGUF/ONNX 需要的额外配置路径(可选) | - |

**示例：**

```bash
# VoxCPM1.5 文字转语音（单个输入）
aha run -m OpenBMB/VoxCPM1.5 -i "太阳当空照" -o output.wav --weight-path /path/to/model

# VoxCPM1.5 从文件读取输入（单个输入）
aha run -m OpenBMB/VoxCPM1.5 -i "file://./input.txt" --weight-path /path/to/model

# MiniCPM4 文本生成（单个输入）
aha run -m OpenBMB/MiniCPM4-0.5B -i "你好" --weight-path /path/to/model

# DeepSeek OCR 图片识别（单个输入）
aha run -m deepseek-ai/DeepSeek-OCR -i "image.jpg" --weight-path /path/to/model

# RMBG2.0 背景移除（单个输入）
aha run -m AI-ModelScope/RMBG-2.0 -i "photo.png" -o "no_bg.png" --weight-path /path/to/model

# GLM-ASR 语音识别（两个输入：提示文本 + 音频文件）
aha run -m ZhipuAI/GLM-ASR-Nano-2512 -i "请转写这段音频" -i "audio.wav" --weight-path /path/to/model

# Fun-ASR 语音识别（两个输入：提示文本 + 音频文件）
aha run -m FunAudioLLM/Fun-ASR-Nano-2512 -i "语音转写：" -i "audio.wav" --weight-path /path/to/model

# qwen3 文本生成（单个输入）
aha run -m Qwen/Qwen3-0.6B -i "你好" --weight-path /path/to/model

# qwen2.5vl 图像理解（两个输入：提示文本 + 图片文件）
aha run -m Qwen/Qwen2.5-VL-3B-Instruct -i "请分析图片并提取所有可见文本内容，按从左到右、从上到下的布局，返回纯文本" -i "image.jpg" --weight-path /path/to/model

# Qwen3-ASR 语音识别（单个输入：音频文件）
aha run -m Qwen/Qwen3-ASR-0.6B -i "audio.wav" --weight-path /path/to/model

# Qwen3.5-GGUF 无mmproj (单个输入：提示文本)
aha run -m qwen3.5-gguf -i 你如何看待AI --gguf-path /path/to/xxx.gguf

# Qwen3.5-GGUF 有mmproj (两个输入：提示文本 + 文件)
aha run -m qwen3.5-gguf -i 提取图片中的文本 -i https://ai.bdstatic.com/file/C56CC9B274CF460CA33
63E59ECD94423 --gguf-path /path/to/xxx.gguf --mmproj-path /path/to/mmproj-xxx.gguf

```

### serv - 启动服务

使用指定模型启动 HTTP 服务。
safetensors模型`--weight-path` 是可选的 - 如果不指定，默认使用 `~/.aha/{model_id}`，
首次启动会从 ModelScope 自动下载（桌宠/本地转换的模型走各自的镜像仓库）。
`qwen3.5-gguf` 不给 `--gguf-path` 时自动下 0.8B Q4_K_M；ONNX 模型必须指定本地文件路径。

**语法：**
```bash
aha serv [OPTIONS] --model <MODEL> [--weight-path <WEIGHT_PATH>] [--gguf-path <GGUF_PATH>] \
  [--mmproj-path <MMPROJ_PATH>] [--onnx-path <ONNX_PATH>] [--config-path <CONFIG_PATH>]
```

**选项：**

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-a, --address <ADDRESS>` | 服务监听地址 | 127.0.0.1 |
| `-p, --port <PORT>` | 服务监听端口 | 10100 |
| `-m, --model <MODEL>` | 模型类型（必选） | - |
| `--weight-path <WEIGHT_PATH>` | 本地模型权重路径（可选） | ~/.aha/{model_id} |
| `--allow-remote-shutdown` | 允许远程关机请求（不推荐） | false |
| `--gguf-path <GGUF_PATH>` | 本地GGUF模型权重路径（使用GGUF模型时必选） | - |
| `--mmproj-path <MMPROJ_PATH>` | 本地mmproj GGUF模型权重路径（可选，未指定则不加载该模块） | - |
| `--onnx-path <ONNX_PATH>` | 本地 ONNX 模型权重 （使用ONNX模型时必选） | - |
| `--config-path <ONNX_PATH>` | GGUF/ONNX 需要的额外配置路径(可选) | - |

**示例：**

```bash
# 使用默认模型路径启动服务 (~/.aha/{model_id})
aha serv -m Qwen/Qwen3-VL-2B-Instruct

# 使用本地模型启动服务
aha serv -m Qwen/Qwen3-VL-2B-Instruct --weight-path /path/to/model

# 指定端口启动
aha serv -m Qwen/Qwen3-VL-2B-Instruct -p 8080

# 指定监听地址
aha serv -m Qwen/Qwen3-VL-2B-Instruct -a 0.0.0.0

# 启用远程关机（不推荐用于生产环境）
aha serv -m Qwen/Qwen3-VL-2B-Instruct --allow-remote-shutdown
```

### ps - 列出运行中的服务

列出所有当前正在运行的 AHA 服务，显示进程 ID、端口和状态。

**语法：**
```bash
aha ps [OPTIONS]
```

**选项：**

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-c, --compact` | 紧凑输出格式（仅显示服务 ID） | false |

**示例：**

```bash
# 列出所有运行中的服务（表格格式）
aha ps

# 紧凑输出（仅服务 ID）
aha ps -c
```

**输出格式：**

```
Service ID           PID        Model                Port       Address         Status
-------------------------------------------------------------------------------------
56860@10100          56860      N/A                  10100      127.0.0.1       Running
```

**字段说明：**
- `Service ID`: 服务唯一标识符，格式为 `pid@port`
- `PID`: 进程 ID
- `Model`: 模型名称（如果未检测到则显示 N/A）
- `Port`: 服务端口号
- `Address`: 服务监听地址
- `Status`: 服务状态（Running、Stopping、Unknown）

### download - 下载模型

仅下载指定模型，不启动服务。

**语法：**
```bash
aha download [OPTIONS] --model <MODEL>
```

**选项：**

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-m, --model <MODEL>` | 模型类型（必选） | - |
| `-s, --save-dir <SAVE_DIR>` | 模型下载保存目录 | ~/.aha/ |
| `--download-retries <DOWNLOAD_RETRIES>` | 下载重试次数 | 3 |

**示例：**

```bash
# 下载模型到默认目录
aha download -m Qwen/Qwen3-VL-2B-Instruct

# 指定保存目录
aha download -m Qwen/Qwen3-VL-2B-Instruct -s /data/models

# 指定下载重试次数
aha download -m Qwen/Qwen3-VL-2B-Instruct --download-retries 5

# 下载 MiniCPM4-0.5B 模型
aha download -m OpenBMB/MiniCPM4-0.5B -s models
```

### delete - 删除已下载的模型

删除默认位置（`~/.aha/{model_id}`）的已下载模型。

**语法：**
```bash
aha delete [OPTIONS] --model <MODEL>
```

**选项：**

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-m, --model <MODEL>` | 模型类型（必选） | - |

**示例：**

```bash
# 删除 RMBG2.0 模型
aha delete -m AI-ModelScope/RMBG-2.0

# 删除 Qwen3-VL-2B 模型
aha delete --model Qwen/Qwen3-VL-2B-Instruct
```

**行为说明：**
- 删除前会显示模型信息（ID、位置、大小）
- 需要用户确认（y/N）才会执行删除
- 如果模型目录不存在，显示"模型未找到"消息
- 删除完成后显示"删除成功"消息

### list - 列出所有支持的模型

列出所有支持的模型及其 ModelScope ID。

**语法：**
```bash
aha list [OPTIONS]
```

**选项：**

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-j, --json` | 以 JSON 格式输出（包含 name、model_id 和 type 字段） | false |

**示例：**

```bash
# 以表格格式列出模型（默认）
aha list

# 以 JSON 格式列出模型
aha list --json

# 简写形式
aha list -j
```

**JSON 输出格式：**

使用 `--json` 时，输出包含：
- `name`：与 `-m` 参数一起使用的模型标识符
- `model_id`：完整的 ModelScope 模型 ID
- `type`：模型类别（`llm`、`ocr`、`asr` 或 `image`）

示例：
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

**模型类型：**
- `llm`：语言模型（文本生成、对话等）
- `ocr`：光学字符识别模型
- `asr`：自动语音识别模型
- `image`：图像处理模型
- `tts`：语音生成

### npu - 探测本机的 NPU 运行时

报告这台机器能给 ONNX Runtime 提供什么：高通 QNN（Hexagon HTP）、NNAPI，以及可执行文件旁边那套 ONNX Runtime 到底编了哪些 execution provider。也会顺带告诉这个 `aha` 构建能不能用 provider。

**语法：**
```bash
aha npu [OPTIONS]
```

**选项：**

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-j, --json` | 以 JSON 格式输出 | false |
| `-r, --recommended` | 只输出建议的 `--provider` 值（没有可用方案时是 `cpu`） | false |

**示例：**

```bash
# 完整报告
aha npu

# 只取要传给 --provider 的值
aha npu -r

# 指向自编的 ONNX Runtime
AHA_NPU_LIB_DIR=/data/local/tmp/qnn-ort aha npu --json
```

**输出**：逐个列出找到的运行时及其路径、从 `libQnnHtpV{N}Stub.so` 解析出的 HTP 版本、本地 ONNX Runtime 内嵌的 execution provider，以及会出什么事（缺 `libQnnSystem.so`、ONNX Runtime 没编 QNN EP、NNAPI 服务已死）。

只有 SenseVoice 和 Kokoro 走 ONNX Runtime，所以只有这两个模型能被 `--provider` 加速。完整说明和需要手工编译的部分见 [NPU 加速](./npu.zh-CN.md)。

### qnn - 检查/执行一个 QAIRT 模型

直接加载一个 `.bin`（由 `scripts/qnn_convert.py --stage` 产出，或任何 QAIRT 转换出来的 context binary），或者一个模型库 `lib<name>.so`（`qnn_convert.py --build-so` 的产物，权重在同目录的 `<name>.bin` 里），把里面的图和张量读出来，可选地跑一次。**它不认识任何具体模型**：图名、张量名、类型、维度全从二进制里读，所以任何模型转出来的产物都能用同一条命令验证。

**语法：**
```bash
aha qnn --context <BIN> [OPTIONS]
aha qnn --model <libNAME.so> [OPTIONS]
```

**选项：**

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--context <BIN>` | context binary 路径（与 `--model` 二选一） | — |
| `--model <SO>` | 模型库路径（与 `--context` 二选一）；权重是同目录的 `<name>.bin` | — |
| `--backend <LIB>` | HTP 后端库 | `libQnnHtp.so` |
| `--system <LIB>` | QNN system 库（读元数据用，仅 `--context` 需要） | `libQnnSystem.so` |
| `--info` | 只打印图和张量，不执行 | false |
| `--graph <N>` | 跑第几张图 | 0 |
| `--input <NAME=PATH>` | 给指定张量喂原始字节，可重复；不给就补零 | — |
| `--out-dir <DIR>` | 把每个输出张量写成 `<名字>.bin` | — |

**示例：**

```bash
# 先看这个 bin 里有什么
aha qnn --context vitsS_ctx.bin --info

# 用真实输入跑一次，输出落盘
aha qnn --context vitsS_ctx.bin \
  --input codes=codes.raw --input text=text.raw \
  --input refer=refer.raw --input sv_emb=sv_emb.raw \
  --out-dir out

# float32 IO 的图（比如 GPT-SoVITS 的 SoVITS）走模型库
aha qnn --model libvitsF.so --info
```

**进程内跑不动时自动回退**：HTP v81 上有些图（IO 是 float32 的 SoVITS 就是）执行时会被 DSP 拒绝，返回 `Qnn_Common_Error_General (1100)`。这种时候 `aha` 会把同样的输入写成 raw 文件 + `input_list.txt`，在产物目录里调用 SDK 自带的 `qnn-net-run`（`--model` 或 `--retrieve_context`），再把 `Result_0/<张量名>.raw` 读回内存，所以命令本身仍然是成功的（回退产物与 `qnn-net-run` 直跑的结果 md5 一致）。前提是产物目录里有 `qnn-net-run`（`scripts/qnn_convert.py --stage` 会带上它），也可以用 `AHA_QNN_NETRUN=<路径>` 指定；`AHA_QNN_NETRUN=0` 关掉回退，用来判断问题出在垫片还是 QNN 运行库。

被 1100 拒过一次之后，同一个进程里的后续调用会直接走回退；`AHA_QNN_SKIP_INPROCESS=1` 连第一次尝试都跳过（省约 0.6 s）。

**用 context binary 还是模型库**：两者都能跑，但模型库路径**每个进程都要 compose + finalize 一次图**，实测 SoVITS 约 **737 s**；同一份图的 context binary 加载只要约 0.7 s、整条链路约 **1.8 s**。所以模型库留给转换/校验（`--info`、喂输入验形状），业务运行请用 context binary（`qnn-context-binary-generator` 生成一次即可）。

**前提**：`aha` 可执行文件旁边要有 `libaha_qnn.so`（`scripts/build_android.py` 在设了 `QNN_SDK_ROOT` 时会自动编译并放在一起），以及设备上的 `libQnnHtp.so` / `libQnnSystem.so`。缺垫片时这条子命令会直接说明缺什么，不影响其它功能。

## 常见使用场景

### 场景 1：快速启动推理服务

```bash
# 一条命令下载并启动服务
aha cli -m Qwen/Qwen3-VL-2B-Instruct
```

### 场景 2：使用已有模型启动服务

```bash
# 假设模型已下载到 /data/models/Qwen/Qwen3-VL-2B-Instruct
aha serv -m Qwen/Qwen3-VL-2B-Instruct --weight-path /data/models/Qwen/Qwen3-VL-2B-Instruct
```

### 场景 3：预先下载模型

```bash
# 下载模型到指定目录，稍后使用
aha download -m Qwen/Qwen3-VL-2B-Instruct -s /data/models

# 后续启动时直接使用
aha serv -m Qwen/Qwen3-VL-2B-Instruct --weight-path /data/models/Qwen/Qwen3-VL-2B-Instruct
```

### 场景 4：自定义服务端口和地址

```bash
# 在 0.0.0.0:8080 启动服务，允许外部访问
aha cli -m Qwen/Qwen3-VL-2B-Instruct -a 0.0.0.0 -p 8080
```

## API 接口

服务启动后，提供以下 API 接口：

### 对话接口
- **端点**: `POST /chat/completions`
- **功能**: 多模态对话和文本生成
- **支持模型**: Qwen2.5VL, Qwen3, Qwen3VL, DeepSeekOCR, GLM-ASR-Nano-2512, Fun-ASR-Nano-2512 等
- **格式**: OpenAI Chat Completion 格式
- **流式支持**: 支持

### 图像处理接口
- **端点**: `POST /images/remove_background`
- **功能**: 图像背景移除
- **支持模型**: RMBG-2.0
- **格式**: OpenAI Chat Completion 格式
- **流式支持**: 不支持

### 语音生成接口
- **端点**: `POST /audio/speech`
- **功能**: 语音合成和生成
- **支持模型**: VoxCPM, VoxCPM1.5
- **格式**: OpenAI Chat Completion 格式
- **流式支持**: 不支持

### Embeddings 接口
- **端点**: `POST /embeddings` 或 `POST /v1/embeddings`
- **功能**: 文本向量生成
- **支持模型**: Qwen3-Embedding 系列
- **格式**: OpenAI embeddings 格式
- **流式支持**: 不支持

### Rerank 接口
- **端点**: `POST /rerank` 或 `POST /v1/rerank`
- **功能**: query/document 重排打分
- **支持模型**: Qwen3-Reranker 系列
- **格式**: Rerank JSON（`results[index,relevance_score,document]`）
- **流式支持**: 不支持

### 关机接口
- **端点**: `POST /shutdown`
- **功能**: 优雅地关闭服务器
- **安全性**: 默认仅允许本地访问，使用 `--allow-remote-shutdown` 标志启用远程访问（不推荐）
- **格式**: JSON 响应


## 获取帮助

```bash
# 查看主帮助
aha --help

# 查看子命令帮助
aha cli --help
aha serv --help
aha download --help

# 查看版本信息
aha --version
```

## 另见

- [快速入门](./getting-started.zh-CN.md) - 快速入门指南
- [NPU 加速](./npu.zh-CN.md) - `--provider`、`aha npu`，以及 NPU 到底能跑什么
- [API 文档](./api.zh-CN.md) - REST API 参考
- [支持的模型](./supported-tools.zh-CN.md) - 可用模型
