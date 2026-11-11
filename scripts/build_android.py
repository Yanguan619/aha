#!/usr/bin/env python3
"""交叉编译 aha 到 Android（aarch64-linux-android）。

    python scripts/build_android.py                    # debug
    python scripts/build_android.py --release          # release
    python scripts/build_android.py --release --features sherpa,gpt-sovits

为什么不用 .ps1：换平台就跑不了。这个脚本在 Windows / Linux / macOS 三个宿主上
都能用（NDK 的 `toolchains/llvm/prebuilt/<host>` 前缀按宿主自动选），只要 Python 3
和 cargo 在。

NDK 未指定时依次找 `ANDROID_NDK_HOME` / `ANDROID_NDK_ROOT`，再找各平台的 SDK 目录
下的 `ndk/`（取版本号最大的一个）。

# 几个必须显式设置的东西（都是踩过的坑）

1. **用裸 clang，并显式传 `--target` / `--sysroot`**：NDK 的
   `aarch64-linux-androidNN-clang` 是批处理/脚本，rustc 传过来的
   `-Wl,--version-script=...` 会被壳重新解析出错。裸 clang 不带默认 target，
   所以 target/sysroot 只能自己给。
2. **`+fp16` 要写在 `CARGO_TARGET_*_RUSTFLAGS` 里**：环境变量会**覆盖**
   `.cargo/config.toml` 的同 target 配置，不带上就丢了。gemm-f16 的内联汇编要它。
3. **`max-page-size=16384`**：Android 15+ 要求 .so 的 LOAD 段 16KB 对齐，
   否则装到手机上会弹"不符合 16 KB 对齐要求"。NDK 27 默认就是，显式写一遍防退化。
4. **CMake 用自带的 Android 支持**（`CMAKE_SYSTEM_NAME=Android`），不用 NDK 的
   legacy toolchain 文件——后者默认 ABI 是 armeabi-v7a，会和 `--target=aarch64` 打架。
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

TRIPLE = "aarch64-linux-android"
ROOT = Path(__file__).resolve().parent.parent


def die(message: str) -> "None":
    print(f"错误：{message}", file=sys.stderr)
    raise SystemExit(1)


def say(message: str) -> None:
    print(f"==> {message}", flush=True)


def host_tag() -> str:
    """NDK 里 `toolchains/llvm/prebuilt/<tag>` 的那一段。

    NDK 至今只发 darwin-x86_64（Apple Silicon 也用它），linux 只有 x86_64。
    """
    if sys.platform.startswith("win"):
        return "windows-x86_64"
    if sys.platform == "darwin":
        return "darwin-x86_64"
    return "linux-x86_64"


def exe(name: str) -> str:
    """按宿主补 .exe 后缀。"""
    return f"{name}.exe" if sys.platform.startswith("win") else name


def find_ndk(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_dir():
            die(f"--ndk 指向的目录不存在：{path}")
        return path

    candidates: list[Path] = []
    for env in ("ANDROID_NDK_HOME", "ANDROID_NDK_ROOT"):
        value = os.environ.get(env)
        if value and Path(value).is_dir():
            candidates.append(Path(value))

    sdk_dirs = [
        os.environ.get("ANDROID_HOME"),
        os.environ.get("ANDROID_SDK_ROOT"),
        # 各平台 SDK 的默认位置
        str(Path.home() / "AppData/Local/Android/Sdk") if sys.platform.startswith("win") else None,
        str(Path.home() / "Library/Android/sdk") if sys.platform == "darwin" else None,
        str(Path.home() / "Android/Sdk"),
        r"D:\Android\Sdk",
    ]
    for sdk in sdk_dirs:
        if not sdk:
            continue
        ndk_dir = Path(sdk) / "ndk"
        if ndk_dir.is_dir():
            # 版本号最大的那个（目录名按字符串排，NDK 的命名天然可比）
            versions = sorted((p for p in ndk_dir.iterdir() if p.is_dir()), key=lambda p: p.name)
            if versions:
                candidates.append(versions[-1])

    if not candidates:
        die(
            "找不到 NDK。用 --ndk <路径> 指定，或设 ANDROID_NDK_HOME"
            "（例如 D:\\Android\\Sdk\\ndk\\26.3.11579264）"
        )
    return candidates[0]


def ensure_rust_std() -> None:
    """确保 target 的 rust-std 装好了；rustup 不行就从 rsproxy 手动拉。"""
    sysroot = subprocess.run(
        ["rustc", "--print", "sysroot"], capture_output=True, text=True, check=True
    ).stdout.strip()
    std_dir = Path(sysroot) / "lib" / "rustlib" / TRIPLE
    if std_dir.is_dir():
        return

    say(f"安装 rust-std（{TRIPLE}）")
    installed = False
    if shutil.which("rustup"):
        # 镜像缺该 target 的 rust-std 时 rustup 会 404，所以看结果而不是看退出码。
        subprocess.run(["rustup", "target", "add", TRIPLE], capture_output=True)
        probe = subprocess.run(
            ["rustc", "--target", TRIPLE, "--print", "target-libdir"],
            capture_output=True,
            text=True,
        )
        installed = probe.returncode == 0 and probe.stdout.strip() != ""

    if not installed:
        # 直接从 rsproxy 拉同一份包手动落盘：版本号和日期都从本地 manifest 里读，
        # 保证和当前 toolchain 完全对齐（拉错版本反而更麻烦）。
        manifest = Path(sysroot) / "lib" / "rustlib" / "multirust-channel-manifest.toml"
        text = manifest.read_text(encoding="utf-8")
        date_match = re.search(r'^date = "(.+)"', text, re.MULTILINE)
        version_out = subprocess.run(
            ["rustc", "-V"], capture_output=True, text=True, check=True
        ).stdout
        if not date_match:
            die(f"读不到 toolchain 日期：{manifest}")
        date = date_match.group(1)
        version = version_out.split()[1]

        url = f"https://rsproxy.cn/dist/{date}/rust-std-{version}-{TRIPLE}.tar.xz"
        say(f"rustup 不可用，改从 {url} 下载")
        with tempfile.TemporaryDirectory(prefix=f"aha-rust-std-{TRIPLE}-") as tmp:
            archive = Path(tmp) / "rust-std.tar.xz"
            try:
                with urllib.request.urlopen(url, timeout=900) as response, archive.open("wb") as out:
                    shutil.copyfileobj(response, out)
            except Exception as error:  # noqa: BLE001 - 网络错误种类很多，统一报
                die(f"下载 rust-std 失败：{url}（{error}）")
            try:
                with tarfile.open(archive, "r:xz") as tar:
                    tar.extractall(tmp)  # noqa: S202 - 只解我们自己刚下的官方包
            except Exception as error:  # noqa: BLE001
                die(f"解压 rust-std 失败：{error}")

            extracted = Path(tmp) / f"rust-std-{version}-{TRIPLE}" / f"rust-std-{TRIPLE}" / "lib" / "rustlib" / TRIPLE
            if not extracted.is_dir():
                die(f"解压后找不到 {extracted}")
            std_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(extracted, std_dir, dirs_exist_ok=True)


def clean_stale_cmake_cache(ndk_dir: Path) -> None:
    """上一次用 VS 生成器留下的缓存会让 cmake 直接报 "generator does not match"。"""
    target_dir = ROOT / "target" / TRIPLE
    if not target_dir.is_dir():
        return
    for cache in target_dir.rglob("CMakeCache.txt"):
        try:
            content = cache.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "CMAKE_GENERATOR:INTERNAL=Visual Studio" in content:
            # build/<crate>-<hash>/out/build/CMakeCache.txt -> 删 out
            build_out = cache.parent.parent
            say(f"清理旧 VS 生成器缓存：{build_out}")
            shutil.rmtree(build_out, ignore_errors=True)


def write_cmake_toolchain(api: int, ndk_dir: Path) -> Path:
    toolchain = Path(tempfile.gettempdir()) / "aha-android-toolchain.cmake"
    toolchain.write_text(
        "set(CMAKE_SYSTEM_NAME Android)\n"
        f"set(CMAKE_SYSTEM_VERSION {api})\n"
        "set(CMAKE_ANDROID_ARCH_ABI arm64-v8a)\n"
        f'set(CMAKE_ANDROID_NDK "{ndk_dir.as_posix()}")\n'
        "set(CMAKE_ANDROID_STL_TYPE c++_static)\n",
        encoding="ascii",
    )
    return toolchain


def read_needed(readelf: Path, artifact: Path) -> str:
    """列出 .so/可执行文件依赖的动态库（没有 llvm-readelf 就算了）。"""
    if not readelf.is_file():
        return "（没有 llvm-readelf，跳过）"
    result = subprocess.run([str(readelf), "-d", str(artifact)], capture_output=True, text=True)
    needed = [
        line.strip()
        for line in result.stdout.splitlines()
        if "NEEDED" in line
    ]
    return ", ".join(needed) if needed else "（无）"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="交叉编译 aha 到 Android（aarch64-linux-android）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "例子：\n"
            "  python scripts/build_android.py\n"
            "  python scripts/build_android.py --release\n"
            "  python scripts/build_android.py --release --features sherpa,gpt-sovits\n"
        ),
    )
    parser.add_argument("--ndk", help="NDK 路径（默认从环境变量 / SDK 目录自动找）")
    parser.add_argument("--api", type=int, default=26, help="目标 API level（默认 26）")
    parser.add_argument("--release", action="store_true", help="release 构建")
    parser.add_argument("--features", help="额外 cargo features，逗号分隔")
    args = parser.parse_args()

    if shutil.which("cargo") is None:
        die("找不到 cargo：先装 Rust 工具链（https://rustup.rs）")
    if shutil.which("ninja") is None:
        die("缺少 ninja（CMake 交叉编译要用它，不能用 VS 生成器）：pip install ninja")

    # ---- 1. NDK ----
    ndk_dir = find_ndk(args.ndk)
    prebuilt = ndk_dir / "toolchains" / "llvm" / "prebuilt" / host_tag()
    llvm_bin = prebuilt / "bin"
    ndk_sysroot = prebuilt / "sysroot"
    clang = llvm_bin / exe("clang")
    clangxx = llvm_bin / exe("clang++")
    if not clang.is_file():
        die(f"NDK 里没有 clang：{clang}（NDK 目录不对，或宿主平台不支持？）")
    say(f"NDK：{ndk_dir}")

    target = f"{TRIPLE}{args.api}"
    clang_flags = f"--target={target} --sysroot={ndk_sysroot}"

    # ---- 2. rust-std ----
    ensure_rust_std()

    # ---- 3. CMake toolchain ----
    cmake_toolchain = write_cmake_toolchain(args.api, ndk_dir)
    clean_stale_cmake_cache(ndk_dir)

    # ---- 4. build ----
    env = os.environ.copy()
    env["PATH"] = str(llvm_bin) + os.pathsep + env.get("PATH", "")
    # 裸 clang 不带默认 target：编译和链接都要显式给。
    env["CC_aarch64_linux_android"] = str(clang)
    env["CXX_aarch64_linux_android"] = str(clangxx)
    env["CFLAGS_aarch64_linux_android"] = clang_flags
    env["CXXFLAGS_aarch64_linux_android"] = clang_flags
    env["AR_aarch64_linux_android"] = str(llvm_bin / exe("llvm-ar"))
    env["CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER"] = str(clangxx)
    # 环境变量会覆盖 .cargo/config.toml 的同 target 配置，所以 +fp16 必须带在这。
    env["CARGO_TARGET_AARCH64_LINUX_ANDROID_RUSTFLAGS"] = (
        f"-C target-feature=+fp16 "
        f"-C link-arg=--target={target} "
        f"-C link-arg=--sysroot={ndk_sysroot} "
        f"-C link-arg=-Wl,-z,max-page-size=16384"
    )
    env["CMAKE_GENERATOR"] = "Ninja"
    env["CMAKE_TOOLCHAIN_FILE"] = str(cmake_toolchain)

    cargo_args = ["cargo", "build", "--target", TRIPLE]
    if args.release:
        cargo_args.append("--release")
    if args.features:
        cargo_args += ["--features", args.features]
    say(" ".join(cargo_args))

    code = subprocess.call(cargo_args, cwd=ROOT, env=env)
    if code != 0:
        die(f"cargo build 失败（exit {code}）")

    # ---- 5. 产物 ----
    profile = "release" if args.release else "debug"
    artifact = ROOT / "target" / TRIPLE / profile / "aha"
    if not artifact.is_file():
        die(f"构建结束但没找到产物：{artifact}")
    size_mb = artifact.stat().st_size / (1024 * 1024)
    say(f"产物：{artifact}（{size_mb:.0f} MB）")
    print(f"    依赖：{read_needed(llvm_bin / exe('llvm-readelf'), artifact)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
