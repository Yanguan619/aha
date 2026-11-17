# NPU Acceleration

How far aha can use a phone's NPU (Qualcomm Hexagon HTP / Apple Neural Engine / Intel NPU), what the `--provider` flag does, and what it takes to get SenseVoice actually running on the HTP.

## Why only some models can use an NPU

aha has two inference paths:

| Path | Models | NPU? |
|------|--------|------|
| candle | LLM, VL, OCR, embeddings, rerank, VLA | No — candle has no mobile NPU backend |
| ONNX Runtime | SenseVoice (ASR), Kokoro (TTS) | Yes, via the QNN execution provider |

So on a phone, `--provider qnn` speeds up speech-to-text and text-to-speech, and nothing else. Everything else keeps running on the CPU no matter what you pass.

On Android 16/17 Google removed the `neuralnetworks` service entirely, so NNAPI — the old vendor-neutral path — no longer has a backend. On modern Snapdragon devices the only real option is Qualcomm's **QNN**, talking to the Hexagon HTP.

## The `--provider` flag

```bash
aha cli --model SenseVoice --provider plugin:/data/local/tmp/qnn/qnn.conf
```

Resolution order:

1. `--provider` (cli / serve / multi / run all accept it)
2. `AHA_PROVIDER` environment variable
3. no provider — ONNX Runtime's default CPU session

Accepted values:

| Value | Meaning |
|-------|---------|
| `cpu` / `auto` | Leave it to ONNX Runtime |
| `qnn` | Qualcomm Hexagon HTP (needs an ONNX Runtime that has the QNN EP built in) |
| `nnapi` | NNAPI (present but dead on Android 16+) |
| `plugin` / `plugin:<conf>` | Use a runtime **plugin EP** (this is how Qualcomm's QNN plugin is loaded) |
| `cuda` | NVIDIA GPU (desktop) |
| `coreml` | Apple Neural Engine / GPU (macOS, iOS) |
| `trt` | TensorRT |
| `rknn` | Rockchip NPU |
| `openvino` | Intel NPU / iGPU |
| `npu` | Shorthand: pick whatever `aha npu -r` finds (`qnn` on Snapdragon, nothing elsewhere) |

Two safety nets:

- If the EP is compiled into ONNX Runtime but cannot initialize (missing driver lib, unsupported op), ONNX Runtime logs it and **silently falls back to CPU**. Watch the log for `Failed to create` / `Fallback to cpu`.
- If the provider you asked for is not the one the probe recommends, aha prints a note on **stderr** at startup (so it shows up even without `RUST_LOG`) instead of leaving you guessing.

`--device npu` is also accepted: on the candle path it maps to the CPU device and logs a warning explaining that the NPU path does not apply to that model. It exists so that a config saying "npu" does not fail to start.

## Probing the device: `aha npu`

```bash
aha npu          # human readable
aha npu --json   # for scripts
aha npu -r       # just the recommended --provider value (cpu when nothing usable)
```

On a Snapdragon 8 Elite Gen 5 (Xiaomi 17 Pro Max, Android 17) it prints (labels are Chinese in the binary; translations in parentheses):

```
平台      android/aarch64              (platform)
QNN       libQnnHtp.so（HTP v81）@ /system_ext/lib64/litert/libQnnHtp.so
          libQnnSystem.so @ /system_ext/lib64/litert/libQnnSystem.so
NNAPI     libneuralnetworks_packageinfo.so
ONNX RT   未找到（不在搜索目录里）        (not found)
其它      libQnnHtpPrepare.so / ...    (other libraries of interest)
  建议    --provider qnn               (recommendation)
```

Searched directories: `AHA_NPU_LIB_DIR` (split on `;` or `:`), the directory of the `aha` binary, then on Android/Linux `/system_ext/lib64/litert`, `/vendor/lib64/hw`, `/vendor/lib64`, `/odm/lib64/hw`, `/odm/lib64`, `/system/lib64`. Point it at a custom ORT build with:

```bash
AHA_NPU_LIB_DIR=/data/local/tmp/qnn-ort aha npu
```

Embedded in an Android app the libraries live in the APK's `lib/arm64-v8a/`; point the probe at `ApplicationInfo.nativeLibraryDir` via `AHA_NPU_LIB_DIR` from the Java side and it will see the ONNX Runtime you shipped.

Two things the report is careful about:

- **NNAPI presence is not availability.** `libneuralnetworks.so` (or its `_packageinfo` stub) can be on disk while the service is gone. The probe refuses to recommend `nnapi` when the ORT next to it has no NNAPI EP.
- **A build without ONNX Runtime gets no recommendation at all.** If aha was compiled without `--features sherpa`, there is nowhere for a provider to go, so `aha npu` says so instead of suggesting `--provider qnn`.

The probe also looks at the **host-side** QAIRT SDK — the one the conversion chain needs, which is a different thing from the runtimes on the device. Set `QNN_SDK_ROOT` and it searches that directory plus its parents for `include/QNN`, `bin/<host>/qnn-onnx-converter`, `lib/aarch64-android` and `share/QNN/converter/jni`:

```
QAIRT SDK   ready @ ...\qairt\2.40.0.251030 (can turn any static ONNX into an HTP model)
```

When something is missing it names the piece (for example, a missing `share/QNN/converter/jni` means "you can produce `.cpp`/`.bin` but cannot build the model library"). QAIRT zips are often unpacked in pieces, so the search walks up and sideways; pointing `QNN_SDK_ROOT` at any one of them is enough.

## Qualcomm's QNN plugin: no QNN SDK needed

Qualcomm ships the QNN backend as a **plugin execution provider** (`onnxruntime-qnn`, on Maven Central as `com.qualcomm.qti:onnxruntime-android-qnn`). It is a single `.so` attached through ONNX Runtime's plugin interface:

```c
env.RegisterExecutionProviderLibrary("qnn", "/path/libonnxruntime_providers_qnn.so");
devices = env.GetEpDevices();                       // filter by EpName()
session_options.AppendExecutionProvider_V2(env, devices, options);
```

The upside is that **the QNN SDK is not required**: at runtime the plugin `dlopen`s the `libQnnHtp.so` / `libQnnHtpPrepare.so` / `libQnnSystem.so` that ship with the device, and the model does not need to be converted to a context binary.

Official plugin releases only cover desktop wheels on PyPI/NuGet; the Android build lives on Maven:

```
https://repo1.maven.org/maven2/com/qualcomm/qti/onnxruntime-android-qnn/
```

The AAR contains exactly one file: `jni/arm64-v8a/libonnxruntime_providers_qnn.so` (~4.4 MB, needs only `liblog/libm/libdl/libc`, libc++ is linked in statically).

sherpa-onnx upstream does not expose this interface yet; the local copy aha builds against does (see below). The configuration is an ini file:

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

## Two requirements a model must satisfy

The HTP does not accept just any ONNX graph:

1. **Static shapes.** The QNN EP does not support dynamic dimensions (`Dynamic shape is not supported yet`), so `x` must be fixed to `[1, frames, 560]`. After LFR, a SenseVoice frame is 60 ms, so 320 frames ≈ 19.2 s.
2. **A quantized model, preferably QDQ.** The HTP is built around fixed-point maths; dynamic quantization operators (`MatMulInteger`, `DynamicQuantizeLinear`) are not in its supported set.

The int8 model in the repo is **dynamically quantized** (281 `DynamicQuantizeLinear` + 281 `MatMulInteger`), which the HTP rejects, so it has to be re-quantized as static QDQ.

### The quantization script

```bash
python scripts/quantize_sense_voice_qnn.py \
  --model-dir E:/weights/sense-voice \
  --num-frames 320 \
  --wav assets/audio/jiangjiang.wav \
  --wav E:/weights/sense-voice/calib-wavs
```

(The model directory is arbitrary; on this machine everything lives under `E:\weights\sense-voice` — put `model.onnx` there and the script writes `model.preproc.onnx` and `model.qnn.onnx` next to it.)

It freezes the shapes with `qnn_preprocess_model`, runs `quantize_static` to produce a QDQ graph, and rewrites the declared `logits` length back to "frames + 4".

**Activations must be 16-bit.** Measured on the same audio, 320-frame window:

| Configuration | logits similarity to fp32 | transcript |
|---------------|---------------------------|------------|
| QUInt8 activations / QUInt8 weights | 0.94 (only 0.18 after the first self-attention block) | all blank |
| **QUInt16 activations / QUInt8 weights** | **0.999** | identical to fp32 |

The reason sits in the attention QK^T: those scores span −202 … 266, and 8-bit per-tensor quantization flattens everything below 1, so softmax ends up nearly uniform. 16-bit activations remove the problem; it is somewhat slower on the HTP, but it is the only combination that works.

### sherpa-onnx side: fixed length + padding

The local sherpa-onnx copy (`third-party/sherpa-onnx-master`) carries three changes:

- `FixedFeatureFrames()` reads the static frame count of `x` from the session input type (-1 for dynamic models).
- `AdjustToFixedFrames()` zero-pads or truncates the LFR features to that count.
- `--provider plugin:<conf>` goes through `GetSharedOrtEnv()` + `GetPluginEpDevices()` + `AppendExecutionProvider_V2`.

> Padding works because the real frame count is kept separately for the decoder. The model itself must be handed the **padded** frame count: it derives its attention mask length from `x_length`, so passing the real length makes the mask disagree with the fixed key length and the `Where` node fails with `Attempting to broadcast an axis by a dimension other than 1. 29 by 324`. The padded tail is silence, decodes to blanks, and does not affect the transcript.

### The native QNN route: any ONNX → HTP model

The plugin route above has a structural weakness: ONNX Runtime decides node support **at session time**, so unsupported ops get pushed back to the CPU and the graph shatters into many QNN partitions (141 for our quantized SenseVoice model). Hit the HTP graph limit and you get `QNN_GRAPH_ERROR_INVALID_HANDLE (6001)`. The native route inverts this: the converter lowers the whole graph to the QNN op set **offline**, failing loudly at conversion time and emitting exactly one graph.

```bash
export QNN_SDK_ROOT=/path/to/qairt/2.40.0.251030
python scripts/qnn_convert.py \
  --onnx E:/weights/gptsovits-onnx/vits64_det.onnx \
  --out-dir E:/weights/gptsovits-qnn --name vitsS \
  --input codes=1,1,64 --input text=1,96 \
  --input refer=1,704,704 --input sv_emb=1,20480 \
  --build-so --stage E:/weights/sovits-stage
```

Four steps in one go:

1. `onnxsim` simplification under the static shapes you pass (required — otherwise nodes like `Pad`/`Reshape` that take initializers break the converter's shape inference);
2. the Linux `qnn-onnx-converter` inside WSL, producing `<name>.cpp` + `<name>.bin`;
3. an Android model library `lib<name>.so`, built with the NDK from that cpp plus QAIRT's `QnnModel.cpp`/`QnnWrapperUtils.cpp`;
4. `--stage` assembles an `adb push`-able directory: model library, `<name>.bin`, the QNN runtime libraries, `qnn-net-run`, `qnn-context-binary-generator`, random `.raw` inputs matching `--input`, and `run.sh`.

On device, `sh run.sh` builds a context binary, then runs `qnn-net-run --num_inferences 20 --profiling_level=basic` and prints wall-clock time (this build of `qnn-net-run` has no `--perf_wall_clock` / `--perf_profile`; passing them exits with `Unused Arguments`). `input_list.txt` is **one line per inference, with the inputs separated by spaces**; writing one input per line is read as "1 input" and fails with `only found input data for 1 inputs from user`.

**Three traps** (handled by the script, but worth knowing):

- `qnn-onnx-converter` **requires numpy<2**. With numpy 2.x, `libPyIrGraph` misreads arrays and you get random garbage dimensions (`IrTensorShape::permute: illegal order [1669419039,5,0]`); even a trivial Transpose fails. Same on Windows and Linux; `numpy==1.26.4` fixes it.
- `libPyIrGraph` is **cp310** only — use Python 3.10 (WSL's Ubuntu-22.04 ships 3.10.12; 3.12/3.13 fail with `ImportError: DLL load failed`).
- The converter's `Pad` translation only accepts activations as buffers; `Pad(initializer)` dies with `KeyError: 'm.enc_p...emb_rel_k'`. That is why step 1 cannot be skipped.

The SDK itself is publicly downloadable (no Qualcomm account): `https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.40.0.251030/v2.40.0.251030.zip` (HEAD returns 403 but a ranged GET returns 206). Do **not** use `curl -C -` to resume it — that corrupts the zip; download in 64 MB ranged chunks instead.

### Running a model from aha: `aha qnn` and `AHA_SOVITS_QNN`

A converted context binary can be inspected and executed with `aha qnn` — it is model agnostic: graph names, tensor names and shapes all come from the binary, nothing is compiled in.

```bash
aha qnn --context vitsS_ctx.bin --info          # list graphs and tensors
aha qnn --context vitsS_ctx.bin \
  --input codes=codes.raw --input text=text.raw \
  --input refer=refer.raw --input sv_emb=sv_emb.raw \
  --out-dir out                                  # run once, writes out/audio.bin
```

It talks to QNN through `libaha_qnn.so` (the whole path is behind the `qnn` cargo feature: on by default in the Android build script, off for desktop builds) (from `native/qnn/aha_qnn.c`, looked up next to the `aha` binary or via `AHA_QNN_LIB`). `scripts/build_android.py` builds that shim automatically when `QNN_SDK_ROOT` is set, and skips it (with a hint, not a failure) when it is not.

GPT-SoVITS' SoVITS decoder is wired to this runtime, switched by an environment variable (a `lib<name>.so` value opens a model library, anything else a context binary).

**Run the context binary, not the model library** — this is the biggest trap we measured:

```bash
# Build vitsS_ctx.bin once with the qnn-context-binary-generator from qnn_convert.py --stage, then reuse it
AHA_SOVITS_QNN=/data/local/tmp/sovits/vitsS_ctx.bin \
AHA_SOVITS_QNN_BACKEND=libQnnHtp.so \
AHA_QNN_LIB=/data/local/tmp/sovits/libaha_qnn.so \
AHA_QNN_SKIP_INPROCESS=1 \
aha run --model gpt-sovits-v2pro-zh --input "你好，世界"
```

Measured on the SM8850 test device (HTP v81), same SoVITS graph:

| Path | Time |
|---|---|
| `AHA_SOVITS_QNN=libvitsF.so` (model library) | **~737 s** — every process composes *and finalises* the graph |
| `AHA_SOVITS_QNN=vitsS_ctx.bin` (context binary, default) | **~2.2 s** |
| same + `AHA_QNN_SKIP_INPROCESS=1` | **~1.8 s** |
| of which one inference (amortised over 20 runs) | **~780 ms** |

The conclusion is unambiguous: **graph preparation** is the entire cost of the model-library path (hundreds of seconds), and a context binary removes it (graphs are pre-finalised, loading is ~0.7 s). Keep the model library for conversion/inspection (`aha qnn --model ... --info`) and run business traffic from the context binary. `AHA_QNN_SKIP_INPROCESS=1` skips the in-process attempt that is known to fail, saving ~0.6 s.

**Why SoVITS does not run in-process**: this graph's IO is all float32, and on HTP v81 `graphExecute` returns `Qnn_Common_Error_General (1100)` (`Dma execution failed on the skel side` on the DSP). This is **not** a "context binary vs model library" difference: both artifact kinds fail the same way in-process (the context binary takes only 0.6 s to fail), while the same artifact runs to completion through the SDK's own `qnn-net-run` (`--model` and `--retrieve_context` both verified). The difference sits in how `qnn-net-run` loads tensors and hands over buffers, which the shim cannot reproduce.

**So `aha` falls back automatically**: when `Runtime::run` sees 1100 it writes the same inputs as raw files plus an `input_list.txt`, runs `qnn-net-run` in the artifact directory (`--model` or `--retrieve_context`), and reads `Result_0/<tensor>.raw` back into memory; the fallback output is **md5-identical** to what `qnn-net-run` produces directly (verified). That needs `qnn-net-run` next to the artifact (`scripts/qnn_convert.py --stage` ships it; `AHA_QNN_NETRUN=<path>` overrides the location). `AHA_QNN_NETRUN=0` disables the fallback, which is how you tell whether a failure comes from the shim or from the QNN runtime. Once a `Runtime` has been refused with 1100, later calls go straight to the fallback instead of repeating the doomed attempt.

**Why one inference is slow, and what this project decided**: QAIRT's profiling aggregate reports `Accelerator (execute)` at ~**359 ms** per inference while wall time is **775 ms** — more than half goes to host-side I/O and RPC waiting, not to DSP compute. What is left is layout movement: every conv in the per-op records is wrapped as
`..._Conv_reshape_to_2d` → `..._reshape_to_2d_nhwc` → `Conv_2d` → `..._intermediate_nchw` → `..._intermediate`, matching **635 Reshape / 618 Transpose / 240 Pad** nodes in the ONNX against only 50 real `ConvTranspose` ops.

Three usual levers were tried and did not help: `--perf_profile high_performance|burst` made no difference (7.7 s vs 7.8 s for 10 runs); `--act-bw 16` and `--act-bw 32` context binaries are effectively identical (7795 ms vs 7725 ms for 10 runs); the 8-bit variant `libvitsQ.so` **cannot even compose** (`QnnModel::addNode() validating node Reshape_0 failed`).

**Decision: the SoVITS decoder runs on CPU (candle) by default in deployment; the NPU path stays available behind `AHA_SOVITS_QNN` for validation and future experiments.** Rationale: 775 ms/run is not better than CPU-class latency and half of it is data movement; a real order-of-magnitude win needs the whole graph reshaped into an NHWC-friendly form (large change plus accuracy re-validation), while the quantization shortcut is blocked by the converter's Reshape handling. Meanwhile end-to-end TTS latency is dominated by the candle-side token-by-token T2S model, so taking the decoder from 775 ms to 50 ms would not change what a user perceives.

If this HTP route is ever restarted, do it in this order, each step independently verifiable: (1) make the graph NHWC-friendly so convolutions transpose once at the boundary instead of per conv, aiming to cut the 1500+ layout ops to single digits; (2) fix the `Reshape_0` quantization encoding so 8-bit can be enabled (halves the bytes moved); (3) split `ref_enc` into its own graph and cache it (it changes once per sentence). Only after all three does "NPU is faster" become a claim worth making.

Convert with `--act-bw 32 --weight-bw 16` (float32 activations); `qnn_convert.py --build-so` then produces `libvitsF.so` + `vitsF.bin`, which must stay in the same directory, and `qnn-context-binary-generator --model libvitsF.so --output_dir . --binary_file vitsF_ctx.bin` turns that into the context binary used at runtime.

**The conversion artifacts are published**: [`Yanguan/gpt-sovits-v2pro-zh-qnn`](https://modelscope.cn/models/Yanguan/gpt-sovits-v2pro-zh-qnn) holds `vitsS_ctx.bin` (154.9 MiB, the one to run), `libvitsS.so` + `vitsS.bin` (for regenerating the context binary), plus a README with sha256, the graph IO table and the measured timings. Fetch it with:

```bash
# CLI
modelscope download --model Yanguan/gpt-sovits-v2pro-zh-qnn --local_dir /data/local/tmp/sovits-qnn
# or from the host / FFI by repo id (no mirror entry → the whole repo)
#   download_model_files("Yanguan/gpt-sovits-v2pro-zh-qnn", Some(dir), 3, DownloadCtl::default())
```

The QNN runtime libraries (`libQnnHtp*.so` / `libQnnSystem.so`) are **not** redistributed there — take them from the QAIRT SDK or use the device's `/system_ext/lib64/litert`.

What did not work (so nobody repeats it): on HTP v81 `QnnMem_register` only accepts `QNN_MEM_TYPE_ION` (`QNN_MEM_TYPE_DMA_BUF` answers `8001` = `QNN_MEM_ERROR_INVALID_HANDLE`), yet feeding the IO as ION memhandles does **not** get around the 1100; `tensorCreateContextTensor` / `tensorCreateGraphTensor` cannot be used on a finalized graph (the first reports `unsupported`, the second only works before `graphAddNode`, i.e. inside the library's compose); wrapping the float32 IO in `DequantizeLinear`/`QuantizeLinear` fails in the converter (`MODEL_GRAPH_OP_VALIDATION_ERROR`, the quantization encoding of that Reshape is undefined); the handle `deviceCreate` returns (it is `0x1` on the test device) must **not** be passed to `contextCreate`, it hangs the DSP inside `graphFinalize` — pass NULL; and the QNN runtime libraries must not be `dlclose`d, otherwise the process jumps into unmapped memory at exit (`qnn-net-run` never unloads them either).

- GPT (T2S) stays on candle — it is token-by-token autoregressive; only `ref_enc + quantizer + enc_p + flow + HiFiGAN` go to the HTP;
- the graph is **fixed shape**: at most 64 code frames (~2.5 s of audio) and 96 phonemes. Longer clauses fall back to candle automatically (with a log line), so only short sentences use the NPU;
- the reference spectrogram follows upstream: first 704 frequency bins (`ref_enc = MelStyleEncoder(704, ...)`), at most 704 frames; the speaker vector is 20480 wide;
- failure reasons are printed to stderr — it never silently degrades.

## Running it on a device

```bash
# 1. push the binary and the libraries
adb push target/aha-npu-bundle/* /data/local/tmp/qnn/
adb shell chmod 755 /data/local/tmp/qnn/aha /data/local/tmp/qnn/ep_probe

# 2. push the model
adb push E:/weights/sense-voice/model.qnn.onnx /data/local/tmp/qnn/models/sense-voice/
adb push E:/weights/sense-voice/model.qnn.onnx.data /data/local/tmp/qnn/models/sense-voice/
adb push E:/weights/sense-voice/tokens.txt /data/local/tmp/qnn/models/sense-voice/

# 3. verify the plugin attaches on its own (no sherpa involved)
adb shell 'cd /data/local/tmp/qnn && LD_LIBRARY_PATH=. ./ep_probe ./libonnxruntime_providers_qnn.so'

# 4. go through aha
adb shell 'cd /data/local/tmp/qnn && LD_LIBRARY_PATH=. \
  AHA_HOME=/data/local/tmp/qnn \
  ./aha cli --model SenseVoice --provider plugin:/data/local/tmp/qnn/qnn.conf --input test.wav'
```

The log should mention `QNNExecutionProvider` rather than `Fallback to cpu`.

## Constraints worth knowing

- **Only SenseVoice and Kokoro.** Any model loaded through candle (LLMs, VLMs, OCR, embeddings) ignores `--provider`.
- **The HTP is a shared block.** It is also used by the camera, the assistant, and system audio. Short bursts keep the NPU from competing with foreground apps; a long-lived session can hold the DSP and make other apps' AI features slower.
- **Fallback is all-or-nothing.** If one operator lacks a QNN kernel, the whole graph runs on CPU; you will not get a partially accelerated session.
- **A fixed frame count is a hard ceiling.** A `[1, 320, 560]` model only accepts audio up to 19.2 s; longer input has to be windowed, and more tiers mean more model copies / QNN graphs.
- **Plugin and ONNX Runtime versions must agree.** The plugin mounts through ORT's C API, so far-apart versions fail to register (the log shows an ORT API version hint).
