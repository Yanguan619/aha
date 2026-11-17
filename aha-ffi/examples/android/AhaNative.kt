package com.example.aha

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import kotlin.concurrent.thread

/**
 * aha 的下载与加载入口（示例封装）。
 *
 * 底层是 `libaha_ffi.so` 的 C ABI，见 `include/aha.h`。这里只包了下载相关的几个调用；
 * 推理（`aha_generate` / `aha_transcribe`）按照同样的方式加 `external fun` 即可。
 *
 * 线程约定：`download` 是挂起函数，内部把阻塞的 `nativeDownload` 丢到专用线程，
 * 调用方协程只负责等结果；进度靠轮询原子计数器，不跨 JNI 回调。
 */
object AhaNative {

    init {
        System.loadLibrary("aha_ffi")
    }

    /** 一次下载的进度快照。 */
    data class Progress(
        val done: Long,
        val total: Long,
        val filesDone: Long,
        val filesTotal: Long,
    ) {
        /** 0f..1f；`total == 0`（还没拿到文件列表）时返回 0。 */
        val fraction: Float
            get() = if (total <= 0L) 0f else (done.toDouble() / total).toFloat()
    }

    class CancelledException : Exception("下载已取消")

    // ---- JNI 桥（包名/类名要和 aha_jni.cpp 里的符号名一致）----
    private external fun nativeModelDir(modelId: String?, saveDir: String?): String
    private external fun nativeDownload(modelId: String?, saveDir: String?, retries: Int): Int
    private external fun nativeCancel()
    private external fun nativeDone(): Long
    private external fun nativeTotal(): Long
    private external fun nativeFilesDone(): Long
    private external fun nativeFilesTotal(): Long
    private external fun nativeLastError(): String

    /** 权重目录：`<saveDir>/<modelId>`；`saveDir` 为 null 时用 `$AHA_HOME` 或 `~/.aha`。 */
    fun modelDir(modelId: String, saveDir: String? = null): String =
        nativeModelDir(modelId, saveDir) ?: throw IllegalStateException(nativeLastError())

    /**
     * 下载模型权重，返回权重目录。
     *
     * 重复调用是安全的：已下完的文件按大小跳过，中途被杀之后直接再调一次就续上。
     *
     * @param modelId `-m` 接收的 id，例如 `qwen3.5-gguf`、`gpt-sovits-v2pro-zh`、`OpenBMB/TurboVLA`。
     * @param saveDir 权重根目录；Android 上建议 `context.filesDir`，传 null 则要求先设 `AHA_HOME`。
     * @param retries 单文件重试次数。
     * @param pollMillis 进度轮询间隔。
     * @throws CancelledException 被 [cancel] 中断。
     */
    suspend fun download(
        modelId: String,
        saveDir: String? = null,
        retries: Int = 3,
        pollMillis: Long = 200,
        onProgress: ((Progress) -> Unit)? = null,
    ): String = withContext(Dispatchers.IO) {
        @Volatile var code = Int.MIN_VALUE
        val worker = thread(name = "aha-download") {
            code = nativeDownload(modelId, saveDir, retries)
        }
        while (worker.isAlive) {
            onProgress?.invoke(readProgress())
            delay(pollMillis)
        }
        worker.join()

        when (code) {
            0 -> modelDir(modelId, saveDir)
            -2 -> throw CancelledException()
            else -> throw IllegalStateException("下载 $modelId 失败：${nativeLastError()}")
        }
    }

    /** 最近的进度（轮询用；`download` 的 `onProgress` 也走这里）。 */
    fun readProgress(): Progress =
        Progress(nativeDone(), nativeTotal(), nativeFilesDone(), nativeFilesTotal())

    /** 请求取消正在进行的 [download]（任意线程可调）。 */
    fun cancel() = nativeCancel()
}
