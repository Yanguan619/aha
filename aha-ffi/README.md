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

## 下载权重

宿主拿到的不只是 `.so`，还要有几百 MB 的权重。`aha_model_download` 把 aha 内置的
ModelScope 下载计划包出来（镜像表见 `src/models/mirrors.rs`），宿主不用知道仓库 id：

```c
static void on_progress(const char *path, uint64_t done, uint64_t total, void *user) {
    printf("\r%s %.1f/%.1f MB", path, done / 1048576.0, total / 1048576.0);
}

/* 阻塞调用：放在你自己的后台线程里，别占 UI 线程。 */
int code = aha_model_download("qwen3.5-gguf", /*save_dir=*/NULL, /*retries=*/3,
                              on_progress, NULL, &err);
if (code == -2) { /* 用户取消（见 aha_model_download_cancel） */ }
if (code == 0) {
    char *dir = aha_model_dir("qwen3.5-gguf", NULL, &err);
    void *model = aha_model_load("qwen3.5-gguf", dir, 4, &err);
    aha_string_free(dir);
}
```

- `save_dir` 传 `NULL` 就是 `~/.aha`；Android 上没有 home，**必须**设 `AHA_HOME` 或显式传目录。
- 不在镜像表里的 id 当成 ModelScope 仓库整仓下载（`aha download -m <repo>` 的语义）。
- 中途被杀没关系：已完成文件按大小跳过，直接再调一次就续上；单文件写 `.part`，校验大小后才改名。
- 不想用回调就传 `NULL`，定期读 `aha_model_download_done()` / `_total()` 画进度条。

Android 的 JNI 壳、Kotlin 封装和 CMake 配置见 [`examples/android/`](examples/android/)。

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
