# Android 接入示例

把 `aha-ffi` 当普通 `.so` 用：Kotlin 侧一个 `object`，C++ 侧一层 JNI 壳，中间是
[`../../include/aha.h`](../../include/aha.h)。这里的东西是**示例**，直接拷进你的 App 即可；
JNI 符号名里的包名（`com.example.aha`）要换成你自己的。

## 目录

| 文件 | 作用 |
|---|---|
| `AhaNative.kt` | Kotlin 封装：`modelDir` / `download` / `cancel` / 进度轮询 |
| `aha_jni.cpp` | JNI 壳：把 Kotlin 调用转成 `aha_*` C 调用，错误转成 Java 异常 |
| `CMakeLists.txt` | 把预编译的 `libaha_ffi.so` 链进 `libaha_jni.so` |

## 1. 交叉编译 libaha_ffi.so

用仓库根目录的脚本（NDK r26，aarch64 + fp16 + 16KB page）：

```bash
python scripts/build_android.py --abi arm64-v8a
# 产物：target/aarch64-linux-android/release/libaha_ffi.so
```

拷到 `app/src/main/jniLibs/arm64-v8a/libaha_ffi.so`（32 位同理再编一个 `armeabi-v7a`）。

> 别用 `target/<triple>/release/libaha_ffi.so` 之外的构建产物：Android 上不能有 glibc 依赖，
> 也不能让 rustls 走平台证书校验（`src/utils/http.rs` 已经改用 webpki-roots，别改回去）。

## 2. 放 JNI 壳

把 `AhaNative.kt`、`aha_jni.cpp` 拷进工程，`build.gradle.kts` 里：

```kotlin
android {
    defaultConfig {
        ndk { abiFilters += listOf("arm64-v8a") }
        externalNativeBuild { cmake { cppFlags += "-std=c++17" } }
    }
    externalNativeBuild { cmake { path = file("src/main/cpp/CMakeLists.txt") } }
}
```

`CMakeLists.txt` 里的 `IMPORTED_LOCATION` 按你的 `jniLibs` 路径改。

## 3. 权重目录：必须显式给

Android 上没有 `$HOME`，aha 默认的 `~/.aha` 落不下去。App 启动时设一次：

```kotlin
val ahaHome = File(filesDir, "aha").apply { mkdirs() }
System.setProperty("AHA_HOME", ahaHome.absolutePath)   // 或在 JNI 里 setenv
```

`AHA_HOME` 是进程级环境变量，**要在第一次调用下载/加载之前设置**；Kotlin 侧也可以直接把
`context.filesDir/aha` 作为 `saveDir` 传给 `AhaNative.download(...)`，两者等价。

## 4. 用

```kotlin
lifecycleScope.launch {
    val dir = AhaNative.download(
        modelId = "qwen3.5-gguf",
        saveDir = File(filesDir, "aha").absolutePath,
        onProgress = { p -> progressBar.progress = (p.fraction * 100).toInt() },
    )
    val handle = AhaNative.loadModel("qwen3.5-gguf", dir)   // 你的加载封装，见 aha.h
}
```

取消：`AhaNative.cancel()`（在别的线程调，下载线程会在下一个 chunk 退出，`download` 抛
`AhaNative.CancelledException`）。

## 5. 几个坑

- **别在主线程调**：`nativeDownload` 是阻塞的（C ABI 没有 async），示例里放在单独线程。
- **进度回调别跨 JNI**：示例走轮询（`nativeDownloadDone/Total`），省掉
  `AttachCurrentThread` 和 JNI 局部引用的问题；真要回调，注意回调线程不是 JVM 线程。
- **首次启动要网络**：镜像在 ModelScope，国内直连；机场/Wi-Fi 切换导致中断时直接重试，
  已完成文件按大小跳过、半截 `.part` 会重下。
- **存储**：0.8B GGUF 约 700 MB，GPT-SoVITS 约 2.4 GB，放 `filesDir` 会算进应用数据；
  大模型建议放 `getExternalFilesDir(null)` 并同样传给 `saveDir`。
