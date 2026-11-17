// JNI 壳：把 Kotlin 的下载调用转成 aha 的 C ABI（见 ../../include/aha.h）。
//
// 符号名里的包名/类名必须和 AhaNative.kt 一致；换包名时这里跟着改（或者拆成
// RegisterNatives 动态注册，示例保持直白）。
//
// 设计取舍：**不从这里抛 Java 异常**。下载跑在 Kotlin 起的工作线程里，JNI 里 ThrowNew
// 会让异常在那个线程上"未捕获"，直接把 App 崩掉；所以失败只返回错误码，错误消息攒在
// lastError 里，由 Kotlin 在 join 之后读到再决定怎么处理。

#include <jni.h>

#include <mutex>
#include <string>

#include "aha.h"

namespace {

std::mutex g_error_mutex;
std::string g_last_error;

void set_last_error(const char* message) {
    std::lock_guard<std::mutex> lock(g_error_mutex);
    g_last_error = message != nullptr ? message : "";
}

std::string to_string(JNIEnv* env, jstring value) {
    if (value == nullptr) {
        return {};
    }
    const char* chars = env->GetStringUTFChars(value, nullptr);
    std::string out = chars != nullptr ? chars : "";
    if (chars != nullptr) {
        env->ReleaseStringUTFChars(value, chars);
    }
    return out;
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_aha_AhaNative_nativeModelDir(JNIEnv* env, jclass, jstring model_id,
                                              jstring save_dir) {
    const std::string id = to_string(env, model_id);
    const std::string dir = to_string(env, save_dir);
    char* err = nullptr;
    char* path = aha_model_dir(id.c_str(), dir.empty() ? nullptr : dir.c_str(), &err);
    if (path == nullptr) {
        set_last_error(err);
        aha_string_free(err);
        return nullptr;
    }
    set_last_error(nullptr);
    jstring out = env->NewStringUTF(path);
    aha_string_free(path);
    return out;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_aha_AhaNative_nativeDownload(JNIEnv* env, jclass, jstring model_id,
                                              jstring save_dir, jint retries) {
    const std::string id = to_string(env, model_id);
    const std::string dir = to_string(env, save_dir);
    char* err = nullptr;
    // 进度走轮询（aha_model_download_done/total），所以回调传 nullptr。
    const int code = aha_model_download(id.c_str(), dir.empty() ? nullptr : dir.c_str(),
                                        retries, /*on_progress=*/nullptr, /*user=*/nullptr, &err);
    if (code == -1) {
        set_last_error(err);
    } else {
        set_last_error(nullptr);
    }
    aha_string_free(err);
    return code;
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_aha_AhaNative_nativeCancel(JNIEnv*, jclass) {
    aha_model_download_cancel();
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_aha_AhaNative_nativeDone(JNIEnv*, jclass) {
    return static_cast<jlong>(aha_model_download_done());
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_aha_AhaNative_nativeTotal(JNIEnv*, jclass) {
    return static_cast<jlong>(aha_model_download_total());
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_aha_AhaNative_nativeFilesDone(JNIEnv*, jclass) {
    return static_cast<jlong>(aha_model_download_files_done());
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_aha_AhaNative_nativeFilesTotal(JNIEnv*, jclass) {
    return static_cast<jlong>(aha_model_download_files_total());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_aha_AhaNative_nativeLastError(JNIEnv* env, jclass) {
    std::lock_guard<std::mutex> lock(g_error_mutex);
    return env->NewStringUTF(g_last_error.c_str());
}
