# aha-ffi

aha 的 C ABI 层：把引擎包成"一个 `.so` + 一个头文件"，任何语言都能直接调。

它**不含任何 JNI**。JNI 的符号名里编着使用方的 Java 包名（`Java_com_sokuji_android_local_AhaNative_*`），
那是消费方的身份，属于消费方的仓库；这一层只提供语言无关的 C 接口，JNI / Swift / Python / Unity
的适配器都长在它上面。

## 构建

```bash
cargo build -p aha-ffi --release              # 只用 candle 后端
cargo build -p aha-ffi --release --features llama   # 再加 llama.cpp（需要 llama-prebuilt/<abi>/）
```

产物：`target/<triple>/release/libaha_ffi.{so,dylib,dll}` 和 `libaha_ffi.a`。

Android 交叉编译沿用 `scripts/build_android.py` 里那套环境变量（见仓库根 README 的 Android 一节），
只是把 `-p aha-ffi --lib` 当作构建目标。

## 用法

头文件：[`include/aha.h`](include/aha.h)

```c
#include "aha.h"

char *err = NULL;
void *model = aha_model_load("glm-asr-nano-2512", "/path/to/asr", 4, &err);
if (!model) { fprintf(stderr, "%s\n", err); aha_string_free(err); return 1; }

char *text = aha_transcribe(model, pcm16, pcm_bytes, 16000,
                            "Please transcribe this audio into text", 256, &err);
if (text) { puts(text); aha_string_free(text); }

aha_model_free(model);
```

流式：

```c
static void on_chunk(const char *t, void *user) { fputs(t, stdout); }
static void on_done(const char *t, void *user) { putchar('\n'); }

AhaCallbacks cb = { on_chunk, on_done, NULL, NULL };
aha_generate_stream(model, messages_json, 128, 0, &cb, &err);
```

## 约定

| 项 | 规则 |
|---|---|
| 句柄 | `void*`，用对应的 `aha_*_free` 释放 |
| 出参字符串 | 归调用方，`aha_string_free` 释放 |
| 入参字符串 | 仍归调用方，库不持有 |
| 错误 | `char** err`，成功时写 `NULL`，失败时写一条 owned 消息 |
| panic | 每个入口 `catch_unwind`，panic 变成错误字符串，绝不跨 FFI |
| 线程 | 句柄内部有 mutex，可在不同线程交替调用；回调在本线程内同步触发 |

## 为什么单独一个包

aha 自己有十几个 `[[bin]]`。如果给 `aha` 本体加 `crate-type = ["cdylib"]`，那么每次 `cargo build`
都要额外链一个 20+ MB 的 `.so`——而做 OCR/VLM 的人根本不需要。所以 C ABI 单独成包，
`cargo build`（不带 `-p`）的行为完全不变。
