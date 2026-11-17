# 更新日志

所有 aha 的重大更改都将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/spec/v2.0.0.html)。

### 2026-11-17
- 新增 `aha npu`：报告这台设备真实可用的 NPU 运行时（QNN HTP 版本、NNAPI、SNPE）以及宿主端
  QAIRT SDK 是否就绪；`--json` / `-r` 便于脚本使用。详见 [npu.zh-CN.md](npu.zh-CN.md)。在 SM8850
  测试机上找到 `/system_ext/lib64/litert/libQnnHtp.so`（HTP v81）
- `--provider`（也认 `AHA_PROVIDER`）：`cpu` / `cuda` / `coreml` / `trt` / `rknn` / `qnn` / `nnapi` /
  `openvino`，或者 `plugin[:<conf>]` 加载运行时插件 EP（比如高通的 QNN 插件）。降级原因一律打到
  stderr，不再静默回退；`--device npu` 会在 candle 侧解析成 CPU（candle 没有移动端 NPU 后端）
- 新增 `aha qnn`：不需要知道模型就能检查/执行一个 QAIRT 模型（图名、张量名、dtype、维度全从产物里读）；
  `--context <bin>` 读 context binary，`--model <libNAME.so>` 组合模型库，`--input NAME=FILE` 喂张量，
  `--out-dir` 写结果。底层是 `native/qnn/aha_qnn.c`（QNN 的接口结构体不能手写，
  必须由 C 编译器保证布局）+ `src/models/common/qnn.rs`
- `aha qnn` / `AHA_SOVITS_QNN` 在进程内执行被 HTP 拒绝时自动回退：`graphExecute` 返回
  `Qnn_Common_Error_General (1100)`（DSP 侧 `Dma execution failed on the skel side`）时，把同样的输入
  写成 raw + `input_list.txt`，在产物目录里调 SDK 自带的 `qnn-net-run`，再把 `Result_0/<张量>.raw`
  读回内存。前提是产物目录里有 `qnn-net-run`（`--stage` 会带上），`AHA_QNN_NETRUN` 可指定路径或设 0 关闭
- 修掉 `native/qnn/aha_qnn.c` 里三个真机上的坑：`QnnModel_freeGraphsInfo` 要的是
  `GraphInfo_t***`（传数组本体时库会把 `graph` 句柄当 `graphName` 去 `free`，进程读 `0x9` 崩）；
  退出时不能 `dlclose` QNN 运行库（否则进程在退出时跳到已卸载的地址上崩）；`deviceCreate` 返回的
  句柄（真机上是 `0x1`）不能传给 `contextCreate`（会让 DSP 卡死在 `graphFinalize`），也别把它当成
  `contextCreateFromBinary` 的 device
- **冷启动**：模型库路径每个进程都要 compose + finalize 一次图，SoVITS 实测 **约 737 s**；
  同一份图的 context binary 已 pre-finalized，加载约 0.7 s、整条链路（含回退）**约 1.8 s**。
  所以模型库只用于转换/校验，业务运行请用 context binary。新增 `AHA_QNN_SKIP_INPROCESS=1` 直接跳
  过注定失败的进程内尝试（省约 0.6 s）；同一个 `Runtime` 被 1100 拒过之后也会自动跳过后续尝试。
  注意单次推理墙钟约 **775 ms**，其中 profiling 聚合的 `Accelerator (execute)` 只有约 **359 ms**，
  其余是主机侧 I/O 与 RPC 等待；per-op 记录显示每个卷积前后都挂着 reshape / NCHW↔NHWC 转换
  （ONNX 侧 635 Reshape / 618 Transpose / 240 Pad 对 50 个 `ConvTranspose`）。int8 变体连构图都
  过不了（`addNode() validating node Reshape_0 failed`），提频与 `act16`/`act32` 均无差别。
  **结论：SoVITS 解码器部署默认走 CPU（candle），`AHA_SOVITS_QNN` 仅作为可选验证路径保留**；
  真要重启 HTP 这条路，顺序是 NHWC 化 → 修 Reshape 量化 → 拆图缓存 `ref_enc`
- `scripts/build_android.py`：设了 `QNN_SDK_ROOT` 就把 `libaha_qnn.so` 编到 `aha` 旁边，没设就提示跳过
  （不算失败）
- 新增 `scripts/qnn_convert.py`：一条命令把任意静态 ONNX 变成 HTP 模型库 + 可 push 的测试包
  （onnxsim → WSL 里的 Linux `qnn-onnx-converter` → NDK 编模型库 → `--stage`）。它还会把权重 `.bin`
  （一堆按张量切好的 `.raw` 打的 tar 包）拆成 object 文件；少了这些，模型库 `dlopen` 会报
  `cannot locate symbol "_binary_obj_binary_*_raw_start"`
- 新增 `scripts/quantize_sense_voice_qnn.py` 与 `scripts/sense_voice_feats.py`：SenseVoice 的定长
  padding + QDQ 路线。QUInt8 激活会把注意力分数压平（logits cos 0.94、全 blank）；QUInt16 激活配
  QUInt8 权重能保住 cos 0.999、文本逐字一致
- SenseVoice 上 HTP：用 k2-fsa 的 SM8850 context binary，sherpa CLI 路径 5.6 秒音频 0.043 秒
  （RTF 0.008），CPU 是 0.173 秒（约 4 倍）；`aha qnn` 跑同一份 bin 只要 15.9 毫秒
- GPT-SoVITS：SoVITS 解码器（文本/参考编码器 + flow + HiFiGAN）可以走 QNN
  （`AHA_SOVITS_QNN=<路径>`，`.so` 按模型库打开、其余按 context binary），自回归的 T2S 仍留在 candle。
  float32 IO 的 context binary 执行会报 `Qnn_Common_Error_General (1100)`（DSP 侧是
  `Dma execution failed on the skel side`，可同一份 bin 用 SDK 自带的 `qnn-net-run` 又能跑），所以
  SoVITS 必须走模型库（`qnn_convert.py --act-bw 32 --weight-bw 16`）；跑不了的路径会自动回退 candle。
  已排除的做法见 [npu.zh-CN.md](npu.zh-CN.md)（HTP v81 不支持 `QnnMem_register` + DMA-BUF；
  `tensorCreateGraphTensor` / `tensorCreateContextTensor` 不能给已 finalized 的图加张量；把 float32 IO
  包成 Q/DQ 会在转换期卡在 Reshape）
- 新增 `scripts/gpt_sovits_export_onnx.py`：把 v2Pro 的 SoVITS 导出成定长 ONNX（codes 64 /
  text 96 / refer 704x704 / sv_emb 20480），与 PyTorch 参考对齐 `cos=1.000000`

### 2026-11-16
- 新增 `/realtime`：对齐 OpenAI Realtime 的 WebSocket 实时语音接口（ASR -> LLM -> TTS），见 [api.zh-CN.md](api.zh-CN.md)
- `multi` 新增 `--llm/--llm-model`：一个进程同时服务对话、ASR、TTS，直接撑起 `/realtime`
- `/realtime` 低延时改造：TTS 改为按块流式返回（首块即出声，不再等整句）、首句 4 字就切、
  `session.enable_thinking`（默认关）、按 `RUST_LOG` 初始化 tracing；配合
  `--device cuda --threads 16`，首音延迟从约 1.9 秒降到 865 毫秒
- `[profile.dev.package."*"] opt-level = 3`：debug 构建也优化依赖（不加时同一轮语音要 160 秒，
  release 只要 0.9 秒）
- 对话模型和 ASR/TTS 一起在启动时预热，`/realtime` 的首轮不再付 CUDA 自动调优与 KV 分配的账
  （实测 1.8 秒 → 常态 0.7 秒）
- 修掉打断/取消响应之后的 `generation_failed: shape mismatch in broadcast_add`：四个生成入口现在
  都会在首次 forward 之前清一次 KV cache，而不是只在流结束时清 —— 流被提前 drop 时尾部的清理
  根本不会执行，下一轮就会拿着上一轮残留的 token 前进，mask 列数对不上
- `turn_detection.silence_duration_ms` 默认值从 500 改为 300：`--threads 16` 下实测回合延迟
  （说完 → 首音）0.97–1.2 秒；压到 250 会把「你好，我是小桌宠，」在逗号停顿处切断
- SenseVoice / Kokoro / Laya 的权重改从 ModelScope 镜像下载（`WEAAEW/...`、`journey0ad/...`、`convaiinnovations/laya-multilingual`），`aha download` 与首次启动都能直接拉取，不再依赖 HF 镜像
- TurboVLA / GPT-SoVITS / qwen3.5-gguf 也放到了 ModelScope：`Yanguan/turbovla-libero`、`Yanguan/gpt-sovits-v2pro-zh`、`Yanguan/qwen3.5-0.8b-gguf`（0.8B Q4_K_M + mmproj-F16），`aha download` / 首次启动自动下载，`--weight-path` / `--gguf-path` 仍可用
- 镜像表收敛到一处（`src/models/mirrors.rs`）：各模型模块、CLI、FFI 下载器都从这里取仓库 id 和必需文件，不再五处各写一份
- `aha-ffi` 导出下载能力：`aha_model_download`（进度回调或轮询、可取消、单文件重试）、`aha_model_dir`、`aha_model_download_cancel`、`aha_model_download_done/total/files_done/files_total`；JNI 壳 + Kotlin 封装在 `aha-ffi/examples/android/`。底层 `src/utils` 新增 `DownloadCtl`（进度/取消钩子，100ms 节流）与 `DownloadSummary`（真正写盘的文件 vs 跳过数）

### 2026-06-06
- 新增 PaddleOCR-VL-1.6

### 2026-05-29
- generate代码重构完成

### 2026-05-28
- generate代码重构进度 1/3

### 2026-05-27
- 新增 MiniCPM5

### 2026-05-24
- 更新文档

### 2026-05-11
- 添加Moss-TTS-Nano，该模型比较小，计算误差影响较大，效果比python原版差

### 2026-05-09
- 合并 pr/eastgold15/46, 添加 aha-ui

### 2026-04-25
- VoxCPM 更新流式生成

### 2026-04-17
- Qwen3ASR 增加 vad 数据识别

### 2026-04-16
- 修复 FireRedVAD fsmn 缓存问题

### 2026-04-15
- 添加 FireRedVAD

### 2026-04-10
- 修复 LiquidAI/LFM2.5-VL-450M chat_template 加载bug

### 2026-04-08
- 添加 VoxCPM2

### 0.2.5 (2026-04-06)
- 添加 qwen3-embedding/qwen3-reranker/all-minilm-l6-v2

### 2026-04-03
- CLI 更新: 必须指定子命令
- ChatCompletionParameters 新增 repeat_penalty 和 repeat_last_n 参数
- 生成添加重复惩罚代码

### 2026-04-02
- 重构生成代码 
- \<think\>...\</think\> 思维链内容使用reasoning_content字段返回。
- 对话返回添加耗时信息

### 2026-04-01
- 重构 deepseek_ocr/fun_asr_nano 生成代码

### 2026-03-31
- 新增 server 和 cli 模块
- aha模型名称使用 modelscope id 替换
- 更新 WhichModel 枚举
- Usage 增加时间信息
- 删除 aha_openai_dive, chrono 依赖

### 2026-03-30
- 新增 LFM2.5VL-1.6B
- 新增 LFM2VL-1.6B

### v0.2.4 (2026-03-23)
- 新增 LFM2.5-1.2B-Instruct
- 新增 LFM2-1.2B

### v0.2.3 (2026-03-18)
- 新增 DeepSeek-OCR-2

### 2026-03-17
- 新增 PaddleOCR-VL1.5 模型
- 修复 qwen3.5 position_ids 创建错误
- cli 参数增加 
  - gguf_path: 本地 GGUF 模型权重路径（加载 GGUF 模型时需要）
  - mmproj_path: 本地 mmproj GGUF 权重路径（加载多模态 GGUF 时需要）
- WhichModel 增加 qwen3.5-gguf

### 2026-03-16
- 增加 Qwen3.5 mmproj

### 2026-03-14
- 更新rust版本 
- 增加了对 Qwen3.5 gguf 的支持，但 4B 模型仍然存在问题；待解决。

## [0.2.2] (2026-03-07)
- 新增 GLM-OCR 模型

## [0.2.1] - (2026-03-05)
- 新增Qwen3.5 模型

### 2026-03-01
- 更新 interpolate.rs

### 2026-02-24
- 更新 candle 版本 0.9.2

## [0.2.0] - 2026-02-05

### 新增
- Qwen3-ASR 语音识别模型

## [0.1.9] - 2026-01-31

### 新增
- CLI `list` 子命令，显示支持的模型
- CLI 子命令结构支持（`cli`、`serv`、`download`、`run`）
- 通过新的 `run` 子命令直接进行模型推理

### 修复
- Qwen3VL thinking startswith bug
- `aha run` 多输入 bug

## [0.1.8] - 2026-01-17

### 新增
- Qwen3 文本模型支持
- Fun-ASR-Nano-2512 语音识别模型

### 修复
- ModelScope Fun-ASR-Nano 模型加载错误

### 变更
- 使用 rubato 更新音频重采样

## [0.1.7] - 2026-01-07

### 新增
- GLM-ASR-Nano-2512 语音识别模型
- Metal (GPU) 支持，适用于 Apple Silicon
- 动态主目录和模型下载脚本

## [0.1.6] - 2025-12-23

### 新增
- RMBG-2.0 背景移除模型
- 图像和音频 API 端点

### 变更
- RMBG2.0 图像处理性能优化

## [0.1.5] - 2025-12-11

### 新增
- VoxCPM1.5 语音生成模型
- PaddleOCR-VL 文字识别模型

## [0.1.4] - 2025-12-09

### 新增
- PaddleOCR-VL 模型支持
- FFmpeg 多媒体处理功能

## [0.1.3] - 2025-12-03

### 新增
- Hunyuan-OCR 模型支持

## [0.1.2] - 2025-11-23

### 新增
- DeepSeek-OCR 模型支持

## [0.1.1] - 2025-11-12

### 新增
- Qwen3-VL 系列模型 (2B, 4B, 8B, 32B)

### 修复
- 为 Qwen3VL 的 tie_word_embeddings 添加 serde 默认值

## [0.1.0] - 2025-10-10

### 新增
- 初始版本发布
- Qwen2.5-VL 模型支持
- MiniCPM4 模型支持
- VoxCPM 语音生成模型
- 兼容 OpenAI 的 REST API
- 所有模型类型的 CLI 界面
