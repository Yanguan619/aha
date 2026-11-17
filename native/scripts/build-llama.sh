#!/usr/bin/env bash
# 把 llama.cpp 编成 Android 静态库，放到 build.rs 期待的 llama-prebuilt/<abi>/。
#
# 普通量化用上游 llama.cpp 即可；Hy-MT2 的 1.25bit / 2bit 是腾讯的 STQ 量化
# （STQ1_0 / STQ_0），必须用带 PR #22836 的源码，否则加载时报
# "unknown quant type"。
#
# 用法（在 Linux / macOS / WSL 里跑，Windows 原生 shell 不行）：
#   export ANDROID_NDK_HOME=$HOME/Android/Sdk/ndk/26.1.10909125
#   export LLAMA_CPP_SRC=$HOME/src/llama.cpp          # 已含 STQ 的源码树
#   native/scripts/build-llama.sh arm64-v8a
#   cargo build --release --features llama --target aarch64-linux-android
#
# 源码准备（GitHub 可访问时）：
#   git clone https://github.com/ggml-org/llama.cpp
#   cd llama.cpp && git fetch origin pull/22836/head:stq && git checkout stq
#
# 源码里还没有 STQ 时，脚本会尝试应用 native/patches/llama-stq.patch
# （或 $LLAMA_STQ_PATCH 指向的文件）。
#
# 另外，腾讯发布的 1.25bit GGUF 把 STQ 数据写在 GGML_TYPE_Q2_0（id 42）上，
# 而 PR 把 STQ 定在 id 43。脚本会再打 native/patches/llama-stq-legacy-id.patch，
# 让 id 42 走 STQ1_0 的实现（否则加载报 tensor offset 不符）。

set -euo pipefail

ABI="${1:-arm64-v8a}"
ANDROID_API="${ANDROID_API:-24}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
llama_src="${LLAMA_CPP_SRC:-$repo_root/../llama.cpp}"
patch_file="${LLAMA_STQ_PATCH:-$repo_root/native/patches/llama-stq.patch}"
legacy_patch="${LLAMA_LEGACY_ID_PATCH:-$repo_root/native/patches/llama-stq-legacy-id.patch}"
out_dir="$repo_root/llama-prebuilt/$ABI"

case "$ABI" in
  arm64-v8a) android_abi=arm64-v8a ;;
  x86_64) android_abi=x86_64 ;;
  armeabi-v7a | arm) android_abi=armeabi-v7a ;;
  *)
    echo "不支持的 ABI: $ABI（可用：arm64-v8a / x86_64 / armeabi-v7a）" >&2
    exit 1
    ;;
esac

if [[ -z "${ANDROID_NDK_HOME:-}" ]]; then
  echo "需要 ANDROID_NDK_HOME（NDK r26+，含 build/cmake/android.toolchain.cmake）" >&2
  exit 1
fi
toolchain="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake"
if [[ ! -f "$toolchain" ]]; then
  echo "找不到 $toolchain，检查 ANDROID_NDK_HOME" >&2
  exit 1
fi
if [[ ! -d "$llama_src" ]]; then
  echo "找不到 llama.cpp 源码：$llama_src（用 LLAMA_CPP_SRC 指定）" >&2
  exit 1
fi

echo "==> llama.cpp: $llama_src"
echo "==> ABI: $android_abi   API: $ANDROID_API   输出: $out_dir"

# STQ 是否已经在源码里：看 ggml 的量化类型定义。
if ! grep -Rqs "STQ1_0\|STQ_0" "$llama_src/ggml"; then
  if [[ -f "$patch_file" ]]; then
    echo "==> 源码不含 STQ，应用 $patch_file"
    git -C "$llama_src" apply --whitespace=nowarn "$patch_file"
  else
    cat >&2 <<'EOF'
源码里没有 STQ 量化，也没有可用的 patch。
1) 拉带 PR #22836 的分支：git fetch origin pull/22836/head:stq && git checkout stq
2) 或者把该 PR 的 diff 存到 native/patches/llama-stq.patch / 用 LLAMA_STQ_PATCH 指定。
EOF
    exit 1
  fi
fi

# 让 GGML_TYPE_Q2_0（id 42）按 STQ1_0 解码：腾讯的 1.25bit GGUF 用的还是旧编号。
if ! grep -A4 '\[GGML_TYPE_Q2_0\] = {' "$llama_src/ggml/src/ggml.c" | grep -q "block_stq1_0"; then
  if [[ -f "$legacy_patch" ]]; then
    echo "==> 应用 type id 42 -> STQ1_0 别名补丁：$legacy_patch"
    git -C "$llama_src" apply --whitespace=nowarn "$legacy_patch"
  else
    echo "警告：找不到 $legacy_patch，官方 1.25bit GGUF 加载时会报 tensor offset 不符" >&2
  fi
fi

build_dir="$llama_src/build-android-$android_abi"
cmake -S "$llama_src" -B "$build_dir" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$toolchain" \
  -DANDROID_ABI="$android_abi" \
  -DANDROID_PLATFORM="android-$ANDROID_API" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF \
  -DGGML_NATIVE=OFF \
  -DGGML_OPENMP=OFF \
  -DGGML_LLAMAFILE=OFF \
  -DGGML_CCACHE=OFF \
  -DLLAMA_CURL=OFF
cmake --build "$build_dir" --config Release -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)" --target llama ggml ggml-cpu ggml-base

mkdir -p "$out_dir"
found=0
for name in llama ggml ggml-cpu ggml-base ggml-blas; do
  for lib in "$build_dir"/lib"$name".a "$build_dir"/src/lib"$name".a "$build_dir"/ggml/src/lib"$name".a; do
    if [[ -f "$lib" ]]; then
      cp -f "$lib" "$out_dir/"
      found=1
      break
    fi
  done
done
if [[ "$found" == 0 ]]; then
  echo "构建完成但没找到 libllama.a / libggml*.a，检查 $build_dir" >&2
  exit 1
fi

# 静态库内部引用 C++ 运行时，Rust 侧按 dylib 链接；从 NDK 里拷现成的。
cxx_so="$(find "$ANDROID_NDK_HOME/toolchains/llvm/prebuilt" -name 'libc++_shared.so' -path "*$android_abi*" | head -n 1 || true)"
if [[ -n "$cxx_so" ]]; then
  cp -f "$cxx_so" "$out_dir/"
else
  echo "警告：没找到 libc++_shared.so，链接阶段可能报缺 c++_shared" >&2
fi

if command -v llvm-nm >/dev/null 2>&1; then
  if llvm-nm "$out_dir/libllama.a" 2>/dev/null | grep -qi "stq"; then
    echo "==> 静态库含 STQ 符号 ✓"
  else
    echo "警告：libllama.a 里没看到 STQ 符号，Hy-MT2 1.25bit 可能仍加载失败" >&2
  fi
fi

echo "==> 完成：$out_dir"
ls -la "$out_dir"
