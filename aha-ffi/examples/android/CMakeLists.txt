# 把预编译的 libaha_ffi.so 链进 JNI 壳。
#
# 目录假设（app/src/main/cpp/ 放本文件时）：
#   app/src/main/cpp/CMakeLists.txt
#   app/src/main/cpp/aha_jni.cpp
#   app/src/main/jniLibs/arm64-v8a/libaha_ffi.so      <- scripts/build_android.py 的产物
#   <aha-ffi>/include/aha.h                            <- 本仓库的 aha-ffi/include
#
# 用 `cargo ndk`/`scripts/build_android.py` 编 libaha_ffi.so 时 target 必须是 aarch64-linux-android，
# 且不能带 glibc 依赖。

cmake_minimum_required(VERSION 3.22)
project(aha_jni CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# aha.h 的位置：按你工程里 aha-ffi 的相对/绝对路径改。
set(AHA_FFI_INCLUDE "${CMAKE_CURRENT_LIST_DIR}/../../../../aha-ffi/include")

add_library(aha_ffi SHARED IMPORTED)
set_target_properties(aha_ffi PROPERTIES
        IMPORTED_LOCATION "${CMAKE_CURRENT_LIST_DIR}/../jniLibs/${ANDROID_ABI}/libaha_ffi.so")

add_library(aha_jni SHARED aha_jni.cpp)
target_include_directories(aha_jni PRIVATE "${AHA_FFI_INCLUDE}")
target_link_libraries(aha_jni aha_ffi log)
