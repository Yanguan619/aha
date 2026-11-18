# 支持的模型

aha 支持多个领域的最先进 AI 模型集合。

```shell
Available models:

Model ID                                 Owner                type       Download  
--------------------------------------------------------------------------------
sentence-transformers/all-MiniLM-L6-v2   sentence-transformers embedding    ✔       
LiquidAI/LFM2-1.2B                       LiquidAI             llm          ✔       
LiquidAI/LFM2.5-1.2B-Instruct            LiquidAI             llm          ✔       
LiquidAI/LFM2.5-VL-1.6B                  LiquidAI             vlm          ✔       
LiquidAI/LFM2-VL-1.6B                    LiquidAI             vlm          ✔       
OpenBMB/MiniCPM4-0.5B                    OpenBMB              llm          ✔   
OpenBMB/MiniCPM5-1B                      OpenBMB              llm          ✔       
Qwen/Qwen2.5-VL-3B-Instruct              Qwen                 vlm          ✔       
Qwen/Qwen2.5-VL-7B-Instruct              Qwen                 vlm                  
Qwen/Qwen3-0.6B                          Qwen                 llm          ✔       
Qwen/Qwen3-1.7B                          Qwen                 llm          ✔       
Qwen/Qwen3-4B                            Qwen                 llm          ✔       
Qwen/Qwen3.5-0.8B                        Qwen                 vlm          ✔       
Qwen/Qwen3.5-2B                          Qwen                 vlm                  
Qwen/Qwen3.5-4B                          Qwen                 vlm                  
Qwen/Qwen3.5-9B                          Qwen                 vlm                  
qwen3.5-gguf                             none                 vlm                  
hy-mt2-1.8b-1.25bit-gguf                 Tencent-Hunyuan      llm                  
Qwen/Qwen3-ASR-0.6B                      Qwen                 asr          ✔       
Qwen/Qwen3-ASR-1.7B                      Qwen                 asr                  
Qwen/Qwen3-Embedding-0.6B                Qwen                 embedding    ✔       
Qwen/Qwen3-Embedding-4B                  Qwen                 embedding            
Qwen/Qwen3-Embedding-8B                  Qwen                 embedding            
Qwen/Qwen3-Reranker-0.6B                 Qwen                 reranker     ✔       
Qwen/Qwen3-Reranker-4B                   Qwen                 reranker             
Qwen/Qwen3-Reranker-8B                   Qwen                 reranker             
Qwen/Qwen3-VL-2B-Instruct                Qwen                 vlm          ✔       
Qwen/Qwen3-VL-4B-Instruct                Qwen                 vlm                  
Qwen/Qwen3-VL-8B-Instruct                Qwen                 vlm                  
Qwen/Qwen3-VL-32B-Instruct               Qwen                 vlm                  
deepseek-ai/DeepSeek-OCR                 deepseek-ai          ocr          ✔       
deepseek-ai/DeepSeek-OCR-2               deepseek-ai          ocr                  
Tencent-Hunyuan/HunyuanOCR               Tencent-Hunyuan      ocr          ✔       
PaddlePaddle/PaddleOCR-VL                PaddlePaddle         ocr          ✔       
PaddlePaddle/PaddleOCR-VL-1.5            PaddlePaddle         ocr
PaddlePaddle/PaddleOCR-VL-1.6            PaddlePaddle         ocr          ✔                  
AI-ModelScope/RMBG-2.0                   AI-ModelScope        image        ✔       
OpenBMB/VoxCPM-0.5B                      OpenBMB              tts          ✔       
OpenBMB/VoxCPM1.5                        OpenBMB              tts          ✔       
ZhipuAI/GLM-ASR-Nano-2512                ZhipuAI              asr          ✔       
FunAudioLLM/Fun-ASR-Nano-2512            FunAudioLLM          asr          ✔       
ZhipuAI/GLM-OCR                          ZhipuAI              ocr          ✔
```

## 文本生成

| 模型 | 参数量 | 模型id | 开源协议 |
|------|--------|------|---------|
| **Qwen3-0.6B** | 0.6B | Qwen/Qwen3-0.6B <br> Qwen/Qwen3-1.7B <br> Qwen/Qwen3-4B | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **MiniCPM4-0.5B** | 0.5B | OpenBMB/MiniCPM4-0.5B | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **MiniCPM5-1B** | 1B | OpenBMB/MiniCPM5-1B | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **LFM2-1.2B** | 1.2B | LiquidAI/LFM2-1.2B | [lfm1.0](https://huggingface.co/LiquidAI/LFM2-1.2B/blob/main/LICENSE) |
| **LFM2.5-1.2B-Instruct** | 1.2B | LiquidAI/LFM2.5-1.2B-Instruct | [lfm1.0](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct/blob/main/LICENSE) |

## 机器翻译

| 模型 | 参数量 | 语言 | 模型id | 开源协议 |
|------|--------|------|--------|---------|
| **Hy-MT2** | 1.8B | 33+ | hy-mt2-1.8b-1.25bit-gguf | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |

> Hy-MT2 走 llama.cpp 而不是 candle：1.25bit 权重是 STQ 量化，架构是自定义的
> `hunyuan-dense`。因此需要 `--features llama` 构建，并先用
> `native/scripts/build-llama.sh arm64-v8a`（Windows 用 `native/scripts/build-llama.ps1`）
> 编出静态库。脚本会打 `native/patches/llama-stq.patch`（llama.cpp PR #22836）和
> `native/patches/llama-stq-legacy-id.patch`，让模型发布时使用的 `Q2_0` 张量类型
> 按 `STQ1_0` 解析。权重从 ModelScope 仓库 `Tencent-Hunyuan/Hy-MT2-1.8B-1.25bit-GGUF`
> 自动下载，也可以用 `--gguf-path` 指向本地文件。模型卡推荐的采样已设为默认值
> （temperature 0.7、top_p 0.6、top_k 20、repetition_penalty 1.05），
> `--ctx-size` 默认 4096。

## Embedding

| 模型 | 参数量 | 模型id | 开源协议 |
|------|--------|------|---------|
| **Qwen3-Embedding** | 0.6B <br> 4B <br> 8B| Qwen/Qwen3-Embedding-0.6B <br> Qwen/Qwen3-Embedding-4B <br> Qwen/Qwen3-Embedding-8B | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **all-MiniLM-L6-v2** | 91M | sentence-transformers/all-MiniLM-L6-v2 | [Apache 2.0](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/main/LICENSE) |

## Reranker

| 模型 | 参数量 | 模型id | 开源协议 |
|------|--------|------|---------|
| **Qwen3-Reranker** | 0.6B <br> 4B <br> 8B| Qwen/Qwen3-Reranker-0.6B <br> Qwen/Qwen3-Reranker-4B <br> Qwen/Qwen3-Reranker-8B | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |

## 视觉与多模态

| 模型 | 参数量 | 模型id | 开源协议 |
|------|--------|------|---------|
| **Qwen2.5-VL** | 3B <br> 7B | Qwen/Qwen2.5-VL-3B-Instruct <br> Qwen/Qwen2.5-VL-7B-Instruct | [Qwen 研究许可协议](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE) <br> [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **Qwen3-VL** | 2B <br> 4B <br> 8B <br> 32B | Qwen/Qwen3-VL-2B-Instruct <br> Qwen/Qwen3-VL-4B-Instruct <br> Qwen/Qwen3-VL-8B-Instruct <br> Qwen/Qwen3-VL-32B-Instruct | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **Qwen3.5** | 0.8B <br> 2B <br> 4B <br> 9B | Qwen/Qwen3.5-0.8B <br> Qwen/Qwen3.5-2B <br> Qwen/Qwen3.5-4B <br> Qwen/Qwen3.5-9B | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **LFM2.5-VL** | 450M <br> 1.6B | LiquidAI/LFM2.5-VL-450M <br> LiquidAI/LFM2.5-VL-1.6B | [lfm1.0](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct/blob/main/LICENSE) |
| **LFM2-VL-1.6B** | 1.6B | LiquidAI/LFM2-VL-1.6B | [lfm1.0](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct/blob/main/LICENSE) |

## OCR

| 模型 | 语言 | 模型id | 开源协议 |
|------|------|------|---------|
| **PaddleOCR-VL** | 多语言 | PaddlePaddle/PaddleOCR-VL <br> PaddlePaddle/PaddleOCR-VL-1.5 <br> PaddlePaddle/PaddleOCR-VL-1.6 | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **Hunyuan-OCR** | 中文 | Tencent-Hunyuan/HunyuanOCR | [腾讯混元社区许可协议](https://huggingface.co/tencent/HunyuanOCR/blob/main/LICENSE) |
| **DeepSeek-OCR** | 多语言 | deepseek-ai/DeepSeek-OCR <br> deepseek-ai/DeepSeek-OCR-2 | [MIT](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md) |
| **GLM-OCR** | 8 | ZhipuAI/GLM-OCR | [MIT](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md) |


## 语音识别 (ASR)

| 模型 | 参数量 | 语言 | 模型id | 开源协议 |
|------|--------|------|-----|---------|
| **Fun-ASR-Nano-2512** | - | 中/英 | FunAudioLLM/Fun-ASR-Nano-2512 | 未标明 |
| **GLM-ASR-Nano-2512** | - | 中/英 | ZhipuAI/GLM-ASR-Nano-2512 | [MIT](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md) |
| **Qwen3-ASR** | 0.6B <br> 1.7B | 中/英 | Qwen/Qwen3-ASR-0.6B <br> Qwen/Qwen3-ASR-1.7B | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **SenseVoice**（sherpa-onnx） | 234M | 中/英/日/韩/粤 | csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17 | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |

> SenseVoice 走 sherpa-onnx（ONNX Runtime）而不是 candle，需要用 `--features sherpa-asr` 构建
> （只想两个后端都带可以做 `--features sherpa`）。权重从 ModelScope 镜像
> `WEAAEW/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17` 自动下载，
> 也可以用 `--weight-path` 指向本地目录。


## 语音生成

| 模型 | 版本 | 模型id | 开源协议 |
|------|--------|------|------|
| **VoxCPM** | 1<br>1.5<br>2 | OpenBMB/VoxCPM-0.5B <br> OpenBMB/VoxCPM1.5 <br> OpenBMB/VoxCPM2 | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| **Kokoro**（sherpa-onnx） | 82M | csukuangfj/kokoro-multi-lang-v1_1 | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |

> Kokoro 走 sherpa-onnx（ONNX Runtime）而不是 candle，需要用 `--features sherpa-tts` 构建。
> 权重从 ModelScope 镜像 `journey0ad/kokoro-multi-lang-v1_1` 自动下载，
> 也可以用 `--weight-path` 指向本地目录。

## 图像处理

| 模型 | 类型 | 模型id | 开源协议 |
|------|------|-----|---------|
| **RMBG-2.0** | 背景移除 | AI-ModelScope/RMBG-2.0 | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.zh-hans) |

## 平台与设备支持矩阵

能跑哪些模型取决于**后端**，不取决于模型本身：candle 家族到处能跑，sherpa / llama.cpp / QNN 各有平台限制。

标记含义：**✅** = 仓库 CI（Linux `ubuntu-latest` + macOS `macos-latest`，跑 `scripts/ci.py` 的
default / sherpa / gpt-sovits / laya / vla / all-no-cuda 组合）或本机实测覆盖了编译 + 测试；
**⚠️** = 代码路径存在，但没在对应平台验证过；**❌** = 明确不支持。

### 后端 × 平台

| 后端 | cargo feature | Windows | Linux | macOS | Android |
|------|---------------|---------|-------|-------|---------|
| candle CPU | 默认 | ✅ 本机 | ✅ CI | ✅ CI | ✅ arm64-v8a（真机）/ armeabi-v7a / x86_64 |
| candle CUDA | `cuda` | ⚠️ 需 NVIDIA 驱动与 CUDA | ⚠️ 同左 | — | — |
| candle Metal | `metal` | — | — | ⚠️ Apple Silicon | — |
| Intel MKL | `mkl` | ⚠️ 仅 x86，首次构建要下约 1 GB | ⚠️ 同左 | — | — |
| sherpa-onnx（ONNX Runtime） | `sherpa-asr` / `sherpa-tts` | ⚠️ 需 `onnxruntime.dll` | ✅ CI | ✅ CI | ✅ 需随包 `libonnxruntime.so` |
| llama.cpp | `llama` | ❌ | ❌ | ❌ | ✅ 仅 Android（`build.rs` 在非 Android 目标直接跳过链接） |
| QNN / Hexagon HTP | `qnn` | ❌ | ❌ | ❌ | ✅ 骁龙（HTP v81 实测），目前只接 SoVITS 解码器 |
| Laya 决策模型 | `laya` | ⚠️ | ✅ CI | ✅ CI | ⚠️ |
| TurboVLA | `vla` | ⚠️ | ✅ CI | ✅ CI | ⚠️ |

`aha npu` / `aha qnn` 两个子命令**只在 Android 构建里存在**：前者是 `#[cfg(target_os = "android")]`，
后者是 `#[cfg(feature = "qnn")]`（Android 构建脚本自动加）。桌面端支持的是 ONNX Runtime 的
`cuda` / `coreml` / `trt` / `openvino` / `plugin:...` provider，用 `--provider` 选，不需要探测。

### 模型族 × 平台

| 模型 | 后端 | Windows | Linux | macOS | Android |
|------|------|---------|-------|-------|---------|
| Qwen3-0.6B / 1.7B / 4B、MiniCPM4-0.5B、MiniCPM5-1B、LFM2-1.2B（llm） | candle | ✅ | ✅ | ✅ | ✅（0.5B–2B 实用） |
| Qwen2.5-VL、Qwen3-VL、Qwen3.5、LFM2-VL（vlm） | candle | ✅ | ✅ | ✅ | ⚠️ 能编能跑，0.8B/2B 可行，4B 起受内存限制 |
| **Hy-MT2 1.25bit（STQ）** | llama.cpp | ❌ | ❌ | ❌ | ✅ 唯一可跑平台 |
| qwen3.5-gguf（Q4_K_M + mmproj，candle 的 gguf 路径） | candle | ✅ | ✅ | ✅ | ✅（0.8B 量化） |
| Qwen3-Embedding / Qwen3-Reranker | candle | ✅ | ✅ | ✅ | ✅（0.6B），4B / 8B 受内存限制 |
| DeepSeek-OCR、PaddleOCR-VL、HunyuanOCR、GLM-OCR | candle | ✅ | ✅ | ✅ | ⚠️ 未在真机验证 |
| RMBG-2.0（image） | candle | ✅ | ✅ | ✅ | ⚠️ 未在真机验证 |
| VoxCPM 0.5B / 1.5 / 2（tts） | candle | ✅ | ✅ | ✅ | ⚠️ 未在真机验证 |
| Kokoro（tts） | sherpa-tts | ⚠️ 需 ORT | ✅ | ✅ | ✅ 需 ORT `.so` |
| SenseVoice（asr） | sherpa-asr | ⚠️ 需 ORT | ✅ | ✅ | ✅ 需 ORT `.so` |
| Qwen3-ASR、GLM-ASR-Nano、Fun-ASR-Nano（asr） | candle | ✅ | ✅ | ✅ | ⚠️ 未在真机验证 |
| GPT-SoVITS v2Pro（tts） | `gpt-sovits`（+ 可选 `qnn`） | ✅ | ✅ | ✅ | ✅ candle 路径；解码器可选走 QNN（见 [npu.zh-CN.md](npu.zh-CN.md)，实测 775 ms/次，默认仍走 CPU） |
| Laya（system1） | `laya` | ⚠️ | ✅ | ✅ | ⚠️ |
| TurboVLA（vla） | `vla` | ⚠️ | ✅ | ✅ | ⚠️ |

> “✅ CI”只说明构建 + 单元测试在那个平台过了，**不等于**每个模型都在那里跑过端到端推理。
> Android 一列的 ✅ 来自真机（SM8850 / Android 17）实测或交叉编译验证。

### Android 侧的硬限制

1. **只有 CPU**：candle 没有移动端 GPU / NPU 后端，`cuda` / `metal` / `mkl` 在 Android 上没有意义。
2. **内存决定能跑多大**：不量化时大致是参数量 × 2 字节，再加 KV cache。0.6B–2B 稳；4B 要 8 GB 以上且不保证；
   8B / 32B 在手机上不现实。想上大模型只能走 GGUF 量化（例如 `qwen3.5-gguf` 的 0.8B Q4_K_M）。
3. **sherpa 模型要随包发 `libonnxruntime.so`**，否则 `SenseVoice` / `Kokoro` 起不来。
   APK 里它位于 `lib/arm64-v8a/`，Java 侧把 `ApplicationInfo.nativeLibraryDir` 设进 `AHA_NPU_LIB_DIR` 即可被找到。
4. **Hy-MT2 必须先编 llama.cpp 静态库**：`native/scripts/build-llama.sh arm64-v8a`（Windows 用 `.ps1`），
   再 `python scripts/build_android.py --features llama`。
5. **QNN 目前只接 SoVITS 解码器**的 float32 图，其余模型没有 QNN 路径；而且实测单次 775 ms（桌面 CPU 参考 330 ms），
   所以部署默认仍走 CPU。

## 模型来源

模型来源：

- [Hugging Face](https://huggingface.co) - 主模型中心
- [ModelScope](https://modelscope.cn) - 中文模型中心

桌宠与本地转换的权重统一托管在 ModelScope，`aha download` / 首次启动自动拉取：

| 模型 | ModelScope 仓库 |
|------|-----------------|
| SenseVoice | `WEAAEW/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17` |
| Kokoro | `journey0ad/kokoro-multi-lang-v1_1` |
| Laya | `convaiinnovations/laya-multilingual` |
| TurboVLA | `Yanguan/turbovla-libero` |
| GPT-SoVITS | `Yanguan/gpt-sovits-v2pro-zh` |
| qwen3.5-gguf | `Yanguan/qwen3.5-0.8b-gguf`（0.8B Q4_K_M + mmproj-F16） |
| Hy-MT2 | `Tencent-Hunyuan/Hy-MT2-1.8B-1.25bit-GGUF`（1.8B 1.25bit STQ GGUF） |

## 添加新模型

参见 [开发指南](./development.zh-CN.md) 了解添加新模型集成的说明。

## 许可证合规

**重要提示**：每个模型都有自己的许可证。在生产环境使用前请查看模型许可证。一些关键注意事项：

- **Apache 2.0**: 宽松许可，支持商业使用
- **MIT**: 宽松许可，支持商业使用
- **Qwen 研究许可协议**: 研究用途，可能有使用限制
- **腾讯混元社区许可协议**: 自定义许可，请查看条款
- **CC BY-NC 4.0**: 仅限非商业用途

在生产环境部署前，请务必验证许可证条款。

## 模型更新

模型不定期更新。

## 性能基准

CPU (M1 Pro) 上的近似推理速度：

| 模型 | 任务 | Tokens/秒 |
|------|------|-----------|
| Qwen3-0.6B | 文本 | 40-50 |
| Qwen2.5-VL-3B | 视觉 | 20-30 |
| Qwen3-ASR-0.6B | ASR | 200-500x |
| VoxCPM-0.5B | TTS | 实时 |

*基准测试因硬件和输入大小而异。*
