# Changelog

All notable changes to aha will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

### 2026-11-17
- add `aha npu`: reports the NPU runtimes this device actually exposes (QNN HTP version, NNAPI,
  SNPE) and whether a host-side QAIRT SDK is usable for converting models, see
  [npu.md](npu.md). On the SM8850 test device it finds `libQnnHtp.so` (HTP v81) under
  `/system_ext/lib64/litert`, while NNAPI no longer has a service on Android 17
- add `--provider`: picks the ONNX Runtime execution provider for sherpa-backed models (`cpu`,
  `cuda`, `coreml`, `trt`, `rknn`, `qnn`, `nnapi`, `openvino`, or `plugin[:<conf>]` for a
  runtime-loaded plugin EP), falling back to `AHA_PROVIDER` and then CPU. Silent downgrades are
  gone: asking for NPU on a machine that cannot run it now says why on stderr instead of quietly
  staying on CPU
- `--device npu` is accepted and resolves to CPU: candle has no mobile NPU backend, so a request for
  NPU only changes the models that go through ONNX Runtime
- add `aha qnn`: inspects and runs a QAIRT model without knowing it (graph names, tensor names,
  dtypes and dims all come from the artifact); `--context <bin>` reads a context binary,
  `--model <libNAME.so>` composes a model library, `--input NAME=FILE` feeds tensors and
  `--out-dir` writes results. Backed by `native/qnn/aha_qnn.c` (a small C shim, because QNN's
  interface structs must not be re-declared by hand) and `src/models/common/runtime/qnn.rs`
- `aha qnn` / `AHA_SOVITS_QNN` fall back automatically when the HTP refuses in-process execution:
  on `Qnn_Common_Error_General (1100)` (`Dma execution failed on the skel side` on the DSP) the
  same inputs are written as raw files plus an `input_list.txt`, the SDK's own `qnn-net-run` runs
  in the artifact directory, and `Result_0/<tensor>.raw` is read back into memory. Needs
  `qnn-net-run` next to the artifact (`--stage` ships it); `AHA_QNN_NETRUN` overrides the path or
  disables the fallback with `0`
- fix three device-only landmines in `native/qnn/aha_qnn.c`:
  `QnnModel_freeGraphsInfo` wants a `GraphInfo_t***` (passing the array itself makes the library
  `free` the graph handle as if it were `graphName`, and the process dies reading `0x9`); the QNN
  runtime libraries must not be `dlclose`d (the process otherwise jumps into unloaded memory at
  exit); and the handle `deviceCreate` returns (`0x1` on the test device) must not be handed to
  `contextCreate`, which hangs the DSP inside `graphFinalize`
- cold start: the model-library path composes *and finalises* the graph in every process — **~737 s**
  for SoVITS — while the same graph's context binary is pre-finalised, loads in ~0.7 s and brings the
  whole chain (including the fallback) down to **~1.8 s**. Keep the model library for
  conversion/inspection and run business traffic from a context binary. `AHA_QNN_SKIP_INPROCESS=1`
  skips the in-process attempt that is known to fail (saves ~0.6 s), and a `Runtime` that was refused
  with 1100 skips it automatically from then on. Wall time per inference is **~775 ms**, of which
  QAIRT's `Accelerator (execute)` aggregate accounts for only ~359 ms — the rest is host-side I/O and
  RPC waiting — and the per-op records show a reshape / NCHW↔NHWC pair around every conv (635 Reshape
  / 618 Transpose / 240 Pad in the ONNX against 50 `ConvTranspose`). The int8 variant cannot even
  compose (`addNode() validating node Reshape_0 failed`), and neither perf profile nor act16/act32
  makes a difference. **Conclusion: the SoVITS decoder runs on CPU (candle) by default in
  deployment; `AHA_SOVITS_QNN` stays as an optional validation path.** Restarting the HTP route means
  NHWC-ifying the graph, fixing the Reshape quantization blocker, then splitting `ref_enc` out
- `scripts/build_android.py` builds `libaha_qnn.so` next to `aha` when `QNN_SDK_ROOT` is set, and
  skips it with a note (not an error) when it is not
- add `scripts/qnn_convert.py`: one command from any static ONNX to an HTP model library plus a
  pushable test bundle (onnxsim → the Linux `qnn-onnx-converter` inside WSL → NDK-compiled model
  library → `--stage`). It also unpacks the weights `.bin` (a tar of per-tensor `.raw` files) into
  object files; without them the model library fails to `dlopen` with `cannot locate symbol
  "_binary_obj_binary_*_raw_start"`
- add `scripts/quantize_sense_voice_qnn.py` and `scripts/sense_voice_feats.py`: the fixed-shape
  padding + QDQ route for SenseVoice. QUInt8 activations collapse the attention scores (logits
  cos 0.94, blank text); QUInt16 activations with QUInt8 weights keep cos 0.999 and identical text
- SenseVoice on HTP: with k2-fsa's SM8850 context binary the sherpa CLI path measures 0.043 s
  (RTF 0.008) for a 5.6 s clip versus 0.173 s on CPU (~4x), and `aha qnn` runs that binary in
  15.9 ms
- GPT-SoVITS: the SoVITS decoder (text/ref encoder + flow + HiFiGAN) can go through QNN via
  `AHA_SOVITS_QNN=<path>`, while the autoregressive T2S stays on candle. A `lib<name>.so` value
  opens a model library, anything else a context binary. Float32-IO context binaries fail
  `graphExecute` with `Qnn_Common_Error_General (1100)` (the DSP reports `Dma execution failed on
  the skel side`, although the SDK's own `qnn-net-run` runs the same bin), so SoVITS needs the
  model library path (`qnn_convert.py --act-bw 32 --weight-bw 16`); anything that cannot run falls
  back to candle automatically — see [npu.md](npu.md) for what has been ruled out
  (`QnnMem_register` + DMA-BUF is unsupported on HTP v81; `tensorCreateGraphTensor` /
  `tensorCreateContextTensor` cannot extend a finalized graph; wrapping float32 IO in Q/DQ fails
  conversion on the Reshape)
- `scripts/gpt_sovits_export_onnx.py`: exports v2Pro's SoVITS to a fixed-shape ONNX (codes 64 /
  text 96 / refer 704x704 / sv_emb 20480), matching the PyTorch reference with `cos=1.000000`

### 2026-11-16
- add `/realtime`: OpenAI Realtime-shaped WebSocket API (ASR -> LLM -> TTS), see [api.md](api.md)
- `multi` gains `--llm/--llm-model`, so one process can serve chat + ASR + TTS for `/realtime`
- `input_audio_buffer.committed` is now sent before the transcription runs (it used to wait for ASR)
- `/realtime` latency work: streaming TTS (audio starts on the first chunk, not after the whole
  sentence), 4-char first sentence, `session.enable_thinking` (off by default), `RUST_LOG`-driven
  tracing init; time-to-first-audio went from ~1.9 s to 865 ms with `--device cuda --threads 16`
- `[profile.dev.package."*"] opt-level = 3`: dependencies are optimized in debug builds (without
  it the same voice turn took 160 s instead of 0.9 s)
- the chat model is warmed at startup alongside ASR/TTS, so the first `/realtime` turn no longer
  pays CUDA autotune + KV allocation (1.8 s → the usual 0.7 s)
- fix `generation_failed: shape mismatch in broadcast_add` after a cancelled/interrupted response:
  every generation entry point now clears the KV cache before the first forward pass, instead of
  only when the token stream ends (an early-dropped stream skipped that cleanup, so the next turn
  ran against a cache still holding the previous turn's tokens)
- `turn_detection.silence_duration_ms` defaults to 300 (was 500); measured turn latency at
  `--threads 16` is 0.97–1.2 s, while 250 cuts 「你好，我是小桌宠，」 at its comma pause
- SenseVoice / Kokoro / Laya weights now come from ModelScope mirrors (`WEAAEW/...`, `journey0ad/...`, `convaiinnovations/laya-multilingual`), so `aha download` and first launch fetch them without an HF mirror
- TurboVLA / GPT-SoVITS / qwen3.5-gguf are on ModelScope too: `Yanguan/turbovla-libero`, `Yanguan/gpt-sovits-v2pro-zh`, `Yanguan/qwen3.5-0.8b-gguf` (0.8B Q4_K_M + mmproj-F16); `aha download` and first launch fetch them automatically, `--weight-path` / `--gguf-path` still work
- the mirror table is now one place (`src/models/mirrors.rs`): each model module, the CLI and the FFI downloader read repo ids + required files from it instead of hard-coding them five times
- `aha-ffi` exports the downloader: `aha_model_download` (progress callback or polling, cancellable, retries per file), `aha_model_dir`, `aha_model_download_cancel`, `aha_model_download_done/total/files_done/files_total`; `aha-ffi/examples/android/` has the JNI shim + Kotlin wrapper, and `DownloadCtl` in `src/utils` carries the progress/cancel hooks (100 ms throttled, `DownloadSummary` reports what was written vs skipped)

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

### 2026-05-09
- merge pr/eastgold15/46, add aha-ui

### 2026-04-25
- VoxCPM update stream

### 2026-04-17
- Qwen3ASR add vad data recognition

### 2026-04-16
- fix FireRedVAD fsmn cache bug

### 2026-04-15
- add FireRedVAD

### 2026-04-10
- fix LiquidAI/LFM2.5-VL-450M chat_template load bug

### 2026-04-08
- add VoxCPM2

### 0.2.5 (2026-04-06)
- add qwen3-embedding/qwen3-reranker/all-minilm-l6-v2

### 2026-04-03
- CLI update: subcommand must be specified
- ChatCompletionParameters add repeat_penalty and repeat_last_n
- generate add penalty repeat

### 2026-04-02
- refactor generate code 
- \<think\>...\</think\> The content of the thought chain is returned using the reasoning_content field.
- response add time info 

### 2026-04-01
- refactor deepseek_ocr/fun_asr_nano generate code 

### 2026-03-31
- add server and cli mod
- aha model name use modelscope id replace 
- update WhichModel enum 
- Usage add time info
- dependencies delete aha_openai_dive,chrono

### 2026-03-30
- add LFM2.5VL-1.6B
- add LFM2VL-1.6B

### v0.2.4 (2026-03-23)
- add LFM2.5-1.2B-Instruct
- add LFM2-1.2B

### v0.2.3 (2026-03-18)
- add DeepSeek-OCR-2

### 2026-03-17
- add PaddleOCR-VL1.5 model
- fix qwen3.5 position_ids create bug
- cli param add 
  - gguf_path: Local GGUF model weight path (required for loading models with GGUF)
  - mmproj_path: Local path to mmproj GGUF weights (required for multimodal model GGUF loading)
- WhichModel add qwen3.5-gguf

### 2026-03-16
- Added Qwen3.5 mmproj

### 2026-03-14
- update rust version 
- Added Qwen3.5 gguf support, but the 4B model still has issues; to be resolved.

## [0.2.2] (2026-03-07)
- Added GLM-OCR model

## [0.2.1] - (2026-03-05)
- Added Qwen3.5 model

### 2026-03-01
- update interpolate.rs

### 2026-02-24
- update candle version 0.9.2

## [0.2.0] - 2026-02-05

### Added
- Qwen3-ASR speech recognition model

## [0.1.9] - 2026-01-31

### Added
- CLI `list` subcommand to show supported models
- CLI subcommand structure support (`cli`, `serv`, `download`, `run`)
- Direct model inference via new `run` subcommand

### Fixed
- Qwen3VL thinking startswith bug
- `aha run` multiple inputs bug

## [0.1.8] - 2026-01-17

### Added
- Qwen3 text model support
- Fun-ASR-Nano-2512 speech recognition model

### Fixed
- ModelScope Fun-ASR-Nano model load error

### Changed
- Updated audio resampling with rubato

## [0.1.7] - 2026-01-07

### Added
- GLM-ASR-Nano-2512 speech recognition model
- Metal (GPU) support for Apple Silicon
- Dynamic home directory and model download script

## [0.1.6] - 2025-12-23

### Added
- RMBG-2.0 background removal model
- Image and audio API endpoints

### Changed
- Performance optimizations for RMBG2.0 image processing

## [0.1.5] - 2025-12-11

### Added
- VoxCPM1.5 voice generation model
- PaddleOCR-VL text recognition model

## [0.1.4] - 2025-12-09

### Added
- PaddleOCR-VL model support
- FFmpeg feature for multimedia processing

## [0.1.3] - 2025-12-03

### Added
- Hunyuan-OCR model support

## [0.1.2] - 2025-11-23

### Added
- DeepSeek-OCR model support

## [0.1.1] - 2025-11-12

### Added
- Qwen3-VL models (2B, 4B, 8B, 32B)

### Fixed
- Added serde default for tie_word_embeddings in Qwen3VL

## [0.1.0] - 2025-10-10

### Added
- Initial release
- Qwen2.5-VL model support
- MiniCPM4 model support
- VoxCPM voice generation model
- OpenAI-compatible REST API
- CLI interface for all model types
