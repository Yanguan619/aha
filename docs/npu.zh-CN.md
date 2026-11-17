# NPU 加速

aha 能用手机 NPU 到什么程度（高通 Hexagon HTP / Apple Neural Engine / Intel NPU）、`--provider` 到底管什么，以及要让 SenseVoice 真的跑在 HTP 上还差哪几步。

## 为什么只有一部分模型能上 NPU

aha 有两条推理路径：

| 路径 | 模型 | NPU？ |
|------|------|-------|
| candle | LLM、VL、OCR、embedding、rerank、VLA | 不行 —— candle 没有移动端 NPU 后端 |
| ONNX Runtime | SenseVoice（语音识别）、Kokoro（语音合成） | 可以，走 QNN execution provider |

所以手机上 `--provider qnn` 加速的是语音识别和语音合成，别的都不动 —— 传什么参数，剩下的模型照样跑 CPU。

Android 16/17 已经整体移除了 `neuralnetworks` 服务，NNAPI 这条老路没有后端了。现代骁龙上现实的选择只有高通的 **QNN**，它把算子交给 Hexagon HTP。

## `--provider` 参数

```bash
aha cli --model SenseVoice --provider plugin:/data/local/tmp/qnn/qnn.conf
```

优先级：

1. `--provider`（cli / serv / multi / run 都认这个参数）
2. 环境变量 `AHA_PROVIDER`
3. 都没有 —— 用 ONNX Runtime 默认的 CPU session

取值：

| 值 | 含义 |
|----|------|
| `cpu` / `auto` | 交给 ONNX Runtime 自己决定 |
| `qnn` | 高通 Hexagon HTP（需要 ONNX Runtime 自带 QNN EP，见下文） |
| `nnapi` | NNAPI（库可能在，Android 16+ 上后端已死） |
| `plugin` / `plugin:<conf>` | 用运行时的 **plugin EP**（高通 QNN plugin 走这条） |
| `cuda` | NVIDIA GPU（桌面） |
| `coreml` | Apple Neural Engine / GPU（macOS、iOS） |
| `trt` | TensorRT |
| `rknn` | 瑞芯微 NPU |
| `openvino` | Intel NPU / 核显 |
| `npu` | 简写：用 `aha npu -r` 探到的那个（骁龙上就是 `qnn`，探不到就什么都不加） |

两道理性保障：

- EP 编进去了但初始化不了（缺驱动库、算子不支持），ONNX Runtime 会打日志然后**静默退回 CPU**。盯住日志里的 `Failed to create` / `Fallback to cpu`。
- 你要的 provider 和探测建议不一致时，aha 会在启动时往 **stderr** 打一行（不设 `RUST_LOG` 也能看见），而不是让你自己猜为什么没变快。

`--device npu` 也认：在 candle 路径上它映射成 CPU 设备并打一条警告，说明这个模型没有 NPU 路径。留着它是为了让写着 "npu" 的配置不会启动失败。

## 探测设备：`aha npu`

```bash
aha npu          # 人看的
aha npu --json   # 给脚本用
aha npu -r       # 只输出建议的 --provider 值（没有可用方案时是 cpu）
```

在一台骁龙 8 Elite Gen 5（小米 17 Pro Max，Android 17）上输出：

```
平台      android/aarch64
QNN       libQnnHtp.so（HTP v81）@ /system_ext/lib64/litert/libQnnHtp.so
          libQnnSystem.so @ /system_ext/lib64/litert/libQnnSystem.so
NNAPI     libneuralnetworks_packageinfo.so
ONNX RT   未找到（不在搜索目录里）
其它      libQnnHtpPrepare.so / libtensorflowlite_jni.so / libadsprpc.so / libcdsprpc.so
  建议    --provider qnn（SenseVoice / Kokoro 走 ONNX Runtime）
```

搜索目录：`AHA_NPU_LIB_DIR`（`;` 或 `:` 分隔）、`aha` 可执行文件所在目录，然后 Android/Linux 上依次是 `/system_ext/lib64/litert`、`/vendor/lib64/hw`、`/vendor/lib64`、`/odm/lib64/hw`、`/odm/lib64`、`/system/lib64`。要指到自编的 ORT：

```bash
AHA_NPU_LIB_DIR=/data/local/tmp/qnn-ort aha npu
```

嵌进 Android 应用时，native 库在 APK 的 `lib/arm64-v8a/` 里，Java 侧可以用 `ApplicationInfo.nativeLibraryDir` 把它设进 `AHA_NPU_LIB_DIR`，探测就能看到随包发的 ONNX Runtime。

报告有两处克制：

- **NNAPI 库在 ≠ 能用。** `libneuralnetworks.so`（或它那个 `_packageinfo` 壳）可能还在盘上，服务早没了。旁边的 ORT 没有 NNAPI EP 时，探测不会推荐 `nnapi`。
- **没编 ONNX Runtime 的构建不下结论。** 如果 aha 没带 `--features sherpa`，provider 没有落点，`aha npu` 会直说，而不是给一句"建议 `--provider qnn`"。

它还会顺带看一眼**宿主侧**的 QAIRT SDK（转换链要用的那一套，和设备上的运行库是两回事）：只要设了 `QNN_SDK_ROOT`，就会从该目录及其上级目录里找 `include/QNN`、`bin/<host>/qnn-onnx-converter`、`lib/aarch64-android`、`share/QNN/converter/jni`，凑齐后打印：

```
QAIRT SDK   就绪 @ ...\qairt\2.40.0.251030（可把任意静态 ONNX 转成 HTP 模型）
```

缺件时直接点名缺哪块（例如只缺 `share/QNN/converter/jni` 时，说明"只能出 .cpp/.bin，编不了模型库"）。QAIRT 的 zip 常被拆成几份解包，探测会往上级目录和兄弟目录里找，所以 `QNN_SDK_ROOT` 指到哪一份都能认出来。

## 用高通 QNN plugin：不用 QNN SDK

Qualcomm 把 QNN 做成了 **plugin execution provider**（`onnxruntime-qnn` 仓库，Maven 上是 `com.qualcomm.qti:onnxruntime-android-qnn`）。它就是一个 `.so`，通过 ONNX Runtime 的 plugin 接口挂进来：

```c
env.RegisterExecutionProviderLibrary("qnn", "/path/libonnxruntime_providers_qnn.so");
devices = env.GetEpDevices();                       // 按 EpName() 过滤
session_options.AppendExecutionProvider_V2(env, devices, options);
```

好处是**不需要 QNN SDK**：plugin 里的 QNN 后端在运行时 `dlopen` 手机系统里的 `libQnnHtp.so` / `libQnnHtpPrepare.so` / `libQnnSystem.so`（骁龙的 Android 系统自带），模型也不需要转成 context binary。

官方发布的 plugin 只有 PyPI/NuGet 上的桌面 wheel，Android 版只在 Maven：

```
https://repo1.maven.org/maven2/com/qualcomm/qti/onnxruntime-android-qnn/
```

AAR 里就一个文件：`jni/arm64-v8a/libonnxruntime_providers_qnn.so`（约 4.4 MB，只依赖 `liblog/libm/libdl/libc`，libc++ 静态链入）。

sherpa-onnx 上游还没接这个接口，aha 用的本地副本加了支持（见下节）。配置写成 ini 文件：

```ini
# qnn.conf
library=/data/local/tmp/qnn/libonnxruntime_providers_qnn.so
ep_name=QNNExecutionProvider
backend_type=htp
backend_path=libQnnHtp.so
htp_arch=81
```

```bash
aha cli --model SenseVoice --provider plugin:/data/local/tmp/qnn/qnn.conf
```

## 模型要满足的两个条件

HTP 不接受"随便什么样"的 ONNX，两条硬门槛：

1. **静态 shape。** QNN EP 明确不支持动态维（`Dynamic shape is not supported yet`），所以 `x` 得固定成 `[1, 帧数, 560]`。SenseVoice 的帧率是 LFR 之后每帧 60 ms，320 帧 ≈ 19.2 秒。
2. **量化模型，而且最好是 QDQ。** HTP 是围绕定点设计的，`MatMulInteger` / `DynamicQuantizeLinear` 这类动态量化算子不在它的支持表里。

仓库里的 int8 模型是**动态量化**的（`DynamicQuantizeLinear` 281 个 + `MatMulInteger` 281 个），直接喂给 HTP 不行，得重做静态 QDQ。

### 量化脚本

```bash
python scripts/quantize_sense_voice_qnn.py \
  --model-dir E:/weights/sense-voice \
  --num-frames 320 \
  --wav assets/audio/jiangjiang.wav \
  --wav E:/weights/sense-voice/calib-wavs
```

（模型目录随意，本机统一放在 `E:\weights\sense-voice`：`model.onnx` 放进去后脚本会就地生成 `model.preproc.onnx` 与 `model.qnn.onnx`。）

它做三件事：`qnn_preprocess_model` 把 shape 固定下来、`quantize_static` 出 QDQ 模型、把 `logits` 的声明长度改回「帧数 + 4」。

**激活必须用 16 位。** 实测 320 帧窗口、同一段音频：

| 配置 | 与 fp32 的 logits 相似度 | 转写结果 |
|------|--------------------------|----------|
| 激活 QUInt8 / 权重 QUInt8 | 0.94（但第一层 self-attn 之后只有 0.18） | 全是 blank，空文本 |
| **激活 QUInt16 / 权重 QUInt8** | **0.999** | 与 fp32 完全一致 |

原因在注意力的 QK^T：那一步的分数范围是 -202 ~ 266，8 位 per-tensor 量化把小于 1 的差异全抹平了，softmax 之后注意力变均匀。16 位激活把这个问题抹掉，代价是 HTP 上略慢，但它是唯一能用的组合。

### sherpa-onnx 侧：定长 + padding

aha 的本地 sherpa-onnx 副本（`C:\...\third-party\sherpa-onnx-master`）加了三个改动：

- `FixedFeatureFrames()`：从 session 的输入类型读出 `x` 的静态帧数（动态模型返回 -1）。
- `AdjustToFixedFrames()`：把 LFR 之后的特征补零或截断到该帧数，`num_frames` 保持真实值。
- `--provider plugin:<conf>` 走 `GetSharedOrtEnv()` + `GetPluginEpDevices()` + `AppendExecutionProvider_V2`。

> 补零靠的是 `x_length` 仍然传真实帧数 —— 但注意 SenseVoice 的 ORT 模型把 `x_length` 参与建 mask，所以**必须传 padding 之后的帧数**，否则 mask 长度和 key 长度对不上，`Where` 节点会报 `Attempting to broadcast an axis by a dimension other than 1. 29 by 324`。补零段落在特征上等价于静音，CTC 解码出来是 blank，不污染结果。

### 原生 QNN 路线：任意 ONNX → HTP 模型（不依赖 ORT 分区）

上面那条 plugin 路线有个硬伤：ORT 是**运行期逐节点判支持**的，不支持的算子会被切回 CPU，图因此被撕成很多个 QNN 分区（我们实测 SenseVoice 的 QDQ 图被切成 141 个），撞上 HTP 的 graph 数量上限就报 `QNN_GRAPH_ERROR_INVALID_HANDLE (6001)`。原生路线反过来：转换器**离线**把整图降到 QNN 算子集，不支持就直接在转换期报错，产出一个 graph。

```bash
export QNN_SDK_ROOT=/path/to/qairt/2.40.0.251030
python scripts/qnn_convert.py \
  --onnx E:/weights/gptsovits-onnx/vits64_det.onnx \
  --out-dir E:/weights/gptsovits-qnn --name vitsS \
  --input codes=1,1,64 --input text=1,96 \
  --input refer=1,704,704 --input sv_emb=1,20480 \
  --build-so --stage E:/weights/sovits-stage
```

一次跑完四步：

1. `onnxsim` 按你给的静态形状简化（必须，否则 `Pad`/`Reshape` 这类带 initializer 的节点会让转换器的 shape 推导崩）；
2. 在 WSL 里调 Linux 版 `qnn-onnx-converter`，产出 `<name>.cpp` + `<name>.bin`；
3. 用 NDK 把 cpp + QAIRT 的 `QnnModel.cpp`/`QnnWrapperUtils.cpp` 编成 Android 模型库 `lib<name>.so`；
4. `--stage` 拼出可直接 `adb push` 的目录：模型库、`<name>.bin`、QNN 运行库（`libQnnHtp*`/`libQnnSystem`）、`qnn-net-run`、`qnn-context-binary-generator`、按 `--input` 生成的随机 raw、`run.sh`。

设备上 `sh run.sh` 会先生成 context binary，再用 `qnn-net-run --num_inferences 20 --profiling_level=basic` 跑 20 次并打印墙钟耗时（这个版本的 `qnn-net-run` 没有 `--perf_wall_clock` / `--perf_profile`，加上会直接报 `Unused Arguments` 退出）。`input_list.txt` 的格式是**一行一次推理、同一行内用空格分隔各个输入**；每个输入单独一行会被理解成"只有 1 个输入"并报 `only found input data for 1 inputs from user`。

**三个已知坑**（脚本里都处理了，自己跑命令时也要注意）：

- `qnn-onnx-converter` **必须 numpy<2**。numpy 2.x 下 `libPyIrGraph` 读数组出错，表现为随机垃圾维度（`IrTensorShape::permute: illegal order [1669419039,5,0]`），连最简单的 Transpose 都必挂；Windows 版和 Linux 版都一样，换 `numpy==1.26.4` 即恢复。
- `libPyIrGraph` 是 **cp310**，得用 Python 3.10（WSL 的 Ubuntu-22.04 自带 3.10.12 正好；3.12/3.13 报 `ImportError: DLL load failed`）。
- `converter` 的 `Pad` translation 只接受 activation 当 buffer，`Pad(initializer)` 会报 `KeyError: 'm.enc_p...emb_rel_k'` —— 所以第 1 步的 onnxsim 不能省。

SDK 本身是公开下载的（不需要 Qualcomm 账号）：`https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.40.0.251030/v2.40.0.251030.zip`（HEAD 返回 403，但 ranged GET 返回 206；**别用 `curl -C -` 续传**，会把 zip 拼坏，要按 64 MB 分块下）。

### 在 aha 里跑模型：`aha qnn` 与 `AHA_SOVITS_QNN`

转换出来的 context binary 可以直接用 `aha qnn` 检查/执行（模型无关：图名、张量名和形状都从 bin 里读，不需要重新编译）：

```bash
aha qnn --context vitsS_ctx.bin --info          # 只打印图与张量
aha qnn --context vitsS_ctx.bin \
  --input codes=codes.raw --input text=text.raw \
  --input refer=refer.raw --input sv_emb=sv_emb.raw \
  --out-dir out                                  # 跑一次，输出写到 out/audio.bin
```

它通过 `native/qnn/aha_qnn.c` 编出来的 `libaha_qnn.so` 访问 QNN（默认在 aha 同目录，找不到时可用 `AHA_QNN_LIB` 指定）。`scripts/build_android.py` 只要设了 `QNN_SDK_ROOT` 就会自动编这个垫片；没设则跳过，`aha qnn` 会给出提示而不是崩溃。

GPT-SoVITS 的 SoVITS 解码链已经接到这个运行时上，用环境变量切换（值是 `lib<name>.so` 时按模型库打开，其余按 context binary）。

**运行请用 context binary，不要用模型库**（这是实测出来的最大坑）：

```bash
# 先用 qnn_convert.py --stage 里的 qnn-context-binary-generator 生成一次 vitsS_ctx.bin，之后一直复用它
AHA_SOVITS_QNN=/data/local/tmp/sovits/vitsS_ctx.bin \
AHA_SOVITS_QNN_BACKEND=libQnnHtp.so \
AHA_QNN_LIB=/data/local/tmp/sovits/libaha_qnn.so \
AHA_QNN_SKIP_INPROCESS=1 \
aha run --model gpt-sovits-v2pro-zh --input "你好，世界"
```

实测耗时（SM8850 / HTP v81，同一份 SoVITS 图）：

| 路径 | 耗时 |
|---|---|
| `AHA_SOVITS_QNN=libvitsF.so`（模型库） | **约 737 s**（每进程都要 compose + finalize 一次图） |
| `AHA_SOVITS_QNN=vitsS_ctx.bin`（context binary，默认） | **约 2.2 s** |
| 同上 + `AHA_QNN_SKIP_INPROCESS=1` | **约 1.8 s** |
| 其中单次推理（20 次连跑摊薄） | **约 780 ms** |

结论很清楚：**图准备**是模型库路径的全部成本（几百秒），context binary 把它压到零（图已 pre-finalized，加载约 0.7 s）。所以模型库路径只用于转换/校验（`aha qnn --model ... --info`），业务运行一律用 context binary。`AHA_QNN_SKIP_INPROCESS=1` 跳过那次注定失败的进程内尝试，省掉约 0.6 s。

**为什么 SoVITS 在进程内跑不动**：这份图的 IO 全是 float32；在 HTP v81 上 `graphExecute` 会返回 `Qnn_Common_Error_General (1100)`，DSP 侧是 `Dma execution failed on the skel side`。这**不是**"context binary 与模型库"的差别 —— 两种产物在进程内都一样失败（context binary 只花 0.6 s 就失败），而同一份产物交给 SDK 自带的 `qnn-net-run` 却能跑完（`--model` 与 `--retrieve_context` 两条路都实测能过）。差异在 `qnn-net-run` 自己的张量装载/缓冲那一层，垫片侧复现不出来。

**所以 `aha` 会自动回退**：`Runtime::run` 遇到 1100 时，把同样的输入写成 raw 文件 + 一个 `input_list.txt`，在产物目录里调 `qnn-net-run`（`--model` 或 `--retrieve_context`），再把 `Result_0/<张量名>.raw` 读回内存。回退产物与 `qnn-net-run` 直跑的产物 **md5 完全一致**（已验证）。前提是产物目录里有 `qnn-net-run`（`scripts/qnn_convert.py --stage` 会带上它，也可以用 `AHA_QNN_NETRUN=<路径>` 指定）；`AHA_QNN_NETRUN=0` 关掉回退，用来确认某次失败到底出在垫片还是 QNN 运行库。同一个 `Runtime` 一旦被 1100 拒过，后面的调用会直接走回退，不再重复那次失败尝试。

**单次推理为什么慢，以及本项目的结论**：QAIRT 的 profiling 聚合项给出 `Accelerator (execute)` ≈ **359 ms/次**，而墙钟是 **775 ms/次** —— 一半以上花在主机侧 I/O 与 RPC 等待，不是 DSP 算力。剩下的那部分又是布局搬运：per-op 记录里每个卷积都被包成
`..._Conv_reshape_to_2d` → `..._reshape_to_2d_nhwc` → `Conv_2d` → `..._intermediate_nchw` → `..._intermediate`，ONNX 侧对应 **635 Reshape / 618 Transpose / 240 Pad**，而真正的 `ConvTranspose` 只有 50 个。

三种常见手段都试过且无效：`--perf_profile high_performance|burst` 与默认无差别（7.7 s vs 7.8 s / 10 次）；`--act-bw 16` 与 `--act-bw 32` 两个 context binary 的耗时几乎相同（7795 ms vs 7725 ms / 10 次）；8-bit 变体 `libvitsQ.so` **连构图都过不了**（`QnnModel::addNode() validating node Reshape_0 failed`）。

**所以本项目的决定是：SoVITS 解码器在部署时默认走 CPU（candle），NPU 路径保留为可选（`AHA_SOVITS_QNN`），用于验证和后续实验。** 理由：775 ms/次不优于 CPU 量级、且一半以上是搬运开销；真要拿到数量级收益需要把整图改成 NHWC 友好形态（属于大改 + 精度复核），而量化那条捷径被转换器的 Reshape 阻塞。相比之下 TTS 的端到端延迟主要由 candle 侧逐 token 自回归的 T2S 决定，先把解码器从 775 ms 压到 50 ms 也不会改变用户感知。

将来要重启 HTP 这条路，按这个顺序做（每步都能独立验证）：① 把图改成 NHWC 友好形态（卷积链只在入口/出口各做一次 transpose），目标是把 1500+ 布局算子压到个位数；② 修掉 `Reshape_0` 的量化编码问题以启用 8-bit（搬运量减半）；③ 把 `ref_enc` 单独拆图并缓存（每句只变一次）。三步都做完再谈"上 NPU 更快"。

转换时用 `--act-bw 32 --weight-bw 16`（float32 激活）即可，`qnn_convert.py --build-so` 会产出 `libvitsF.so` + `vitsF.bin`（两个文件必须同目录），再由 `qnn-context-binary-generator --model libvitsF.so --output_dir . --binary_file vitsF_ctx.bin` 生成运行时用的 context binary。

排查结论（省得重复踩）：`QnnMem_register` 在 HTP v81 上只认 `QNN_MEM_TYPE_ION`（`QNN_MEM_TYPE_DMA_BUF` 直接回 `8001` = `QNN_MEM_ERROR_INVALID_HANDLE`），但用 ION memhandle 喂 IO 也**没能**绕过 1100；`tensorCreateContextTensor` / `tensorCreateGraphTensor` 对 finalized 的图不可用（前者报 `unsupported`，后者只能加在 `graphAddNode` 之前，也就是模型库的 compose 内部）；把 float32 IO 用 `DequantizeLinear/QuantizeLinear` 包成 uint16 也不行（转换器会在该 Reshape 上报 `MODEL_GRAPH_OP_VALIDATION_ERROR`，因为量化编码未定义）；`deviceCreate` 返回的句柄（真机上是 `0x1`）**不要**传给 `contextCreate`，会让 DSP 卡死在 `graphFinalize`，传 NULL 才正常；QNN 运行库也不要 `dlclose`，否则进程退出时会跳到已卸载的数据地址上崩掉（`qnn-net-run` 同样不卸载）。

- GPT(T2S) 仍是 candle 逐 token 自回归，只有 `ref_enc + quantizer + enc_p + flow + HiFiGAN` 这一段进 HTP；
- 图是**定长**的：codes 最多 64 帧（约 2.5 s 音频）、text 96 个音素；超出的句子自动回退 candle（日志里有说明），所以短句才走 NPU；
- 参考谱按官方约定取前 704 个频点（`ref_enc = MelStyleEncoder(704, ...)`）、最多 704 帧；说话人向量 20480 维；
- 打开的失败原因会打印到 stderr，不会静默降级。

## 真机步骤

```bash
# 1. 送二进制和库
adb push target/aha-npu-bundle/* /data/local/tmp/qnn/
adb shell chmod 755 /data/local/tmp/qnn/aha /data/local/tmp/qnn/ep_probe

# 2. 送模型
adb push E:/weights/sense-voice/{model.qnn.onnx,model.qnn.onnx.data,tokens.txt} /data/local/tmp/qnn/models/sense-voice/

# 3. 先单独验证 plugin 能挂上（不经过 sherpa）
adb shell 'cd /data/local/tmp/qnn && LD_LIBRARY_PATH=. ./ep_probe ./libonnxruntime_providers_qnn.so'

# 4. 走 aha
adb shell 'cd /data/local/tmp/qnn && LD_LIBRARY_PATH=. \
  AHA_HOME=/data/local/tmp/qnn \
  ./aha cli --model SenseVoice --provider plugin:/data/local/tmp/qnn/qnn.conf --input test.wav'
```

日志里该出现 `QNNExecutionProvider`，而不是 `Fallback to cpu`。

## 几条硬约束

- **只有 SenseVoice 和 Kokoro。** 走 candle 的模型（LLM、VLM、OCR、embedding）完全不理 `--provider`。
- **HTP 是共享硬件。** 相机、语音助手、系统音频都在用它。短脉冲式推理能少抢资源；常驻 session 会占住 DSP，让别的应用的 AI 功能变慢。
- **回退是全有或全无。** 只要有一个算子没有 QNN kernel，整张图就跑 CPU，不会给你"部分加速"的 session。
- **固定帧数就是上限。** `[1, 320, 560]` 的模型只吃 19.2 秒以内的音频，更长的要切窗；想要更多档位就得多份模型/QNN graph。
- **plugin 和 ORT 版本要对得上。** plugin 用 ORT 的 C API 挂载，工具链版本差太远会直接注册失败（日志里能看到 ORT API 版本提示）。
