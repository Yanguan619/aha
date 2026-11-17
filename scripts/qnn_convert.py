#!/usr/bin/env python3
"""把任意静态 shape 的 ONNX 转成 Qualcomm QNN (HTP) 能跑的模型。

这个脚本是「模型无关」的那一层：它不认识 SenseVoice 也不认识 SoVITS，
只要求调用方把输入形状钉死，剩下的交给官方 QAIRT 转换器。
SenseVoice 那份带校准的 QDQ 量化在 ``quantize_sense_voice_qnn.py``，
本脚本吃的是它的产物（已经是静态 shape 的 ONNX）。

流程（和官方 QAIRT 一致）：

1. ``onnxsim`` 把 ``Shape``/``Gather``/``Pad(initializer)`` 这类动态图元折掉 ——
   转换器的 C++ shape 推导遇到它们会直接崩（``KeyError: m...emb_rel_k``、
   ``Unable to calculate ReshapeOp output shape``）；
2. ``qnn-onnx-converter`` 把 ONNX 降成 QNN 模型源码（``<name>.cpp``）+ 权重（``<name>.bin``）；
3. 可选 ``--build-so``：用 NDK 把生成的源码编译成 Android 模型库 ``lib<name>.so``。

两个实测坑（都会让人误以为是"转换器坏了"）：

* ``qnn-onnx-converter`` **必须配 numpy<2**。numpy 2.x 下 ``libPyIrGraph`` 读数组出错，
  表现为随机垃圾维度（``IrTensorShape::permute: illegal order [1669419039,5,0]``）。
* ``libPyIrGraph`` 是 **cp310** 专用，宿主 python 必须是 3.10。

用法::

    # 只看要执行什么，不真的跑
    python scripts/qnn_convert.py --onnx model.onnx --out-dir build \\
        --input codes=1,1,64 --input text=1,96 --dry-run

    # 真跑（Windows 上会自动进 WSL，Linux 上直接执行）
    python scripts/qnn_convert.py --onnx E:/weights/gptsovits-onnx/vits64_det.onnx \\
        --out-dir E:/weights/gptsovits-qnn --name vitsS \\
        --input codes=1,1,64 --input text=1,96 \\
        --input refer=1,704,704 --input sv_emb=1,20480 --build-so

产物：``<out-dir>/<name>.cpp``、``<name>.bin``（以及可选的 ``lib<name>.so``）。
``.<name>.meta.json`` 记录输入形状、位宽与各步骤耗时，方便复现。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tarfile
import sys
import time
from pathlib import Path


# QAIRT 转换器默认位置（解包后的目录结构）：
#   <root>/bin/<platform>/qnn-onnx-converter
#   <root>/lib/python                    ← PYTHONPATH
#   <root>/lib/<platform>                ← LD_LIBRARY_PATH
PLATFORMS = ("x86_64-linux-clang", "x86_64-windows-msvc")


def parse_inputs(values: list[str]) -> list[tuple[str, str]]:
    """把 ``name=1,1,64`` 解析成 ``[("name", "1,1,64")]``。"""
    out: list[tuple[str, str]] = []
    for raw in values:
        if "=" not in raw:
            raise SystemExit(f"--input 需要 name=dims 形式，收到 {raw!r}")
        name, dims = raw.split("=", 1)
        name = name.strip()
        dims = dims.strip()
        if not name or not dims:
            raise SystemExit(f"--input 的 name/dims 不能为空：{raw!r}")
        for part in dims.split(","):
            if not part.strip().isdigit() or int(part) <= 0:
                raise SystemExit(
                    f"维度必须是正整数字面量（这是 QNN 的硬要求）：{raw!r}"
                )
        out.append((name, dims))
    return out


def find_sdk_root(explicit: str | None) -> Path:
    """按 参数 → QNN_SDK_ROOT 的顺序找 QAIRT 根目录。"""
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("QNN_SDK_ROOT"):
        candidates.append(Path(os.environ["QNN_SDK_ROOT"]))
    for cand in candidates:
        if (cand / "lib" / "python").is_dir():
            return cand.resolve()
    raise SystemExit(
        "找不到 QAIRT SDK 根目录（要含 lib/python）。"
        "用 --sdk-root 指定，或设置 QNN_SDK_ROOT。"
    )


def find_converter(root: Path, platform: str) -> Path:
    path = root / "bin" / platform / "qnn-onnx-converter"
    if not path.exists():
        raise SystemExit(f"这个 SDK 里没有 converter：{path}")
    return path


def to_wsl_path(path: Path) -> str:
    """``E:\\weights\\x`` → ``/mnt/e/weights/x``（非 Windows 原样返回）。"""
    resolved = path.resolve()
    if os.name != "nt":
        return str(resolved)
    drive = resolved.drive.rstrip(":").lower()
    rest = str(resolved)[len(resolved.drive) :].replace("\\", "/").lstrip("/")
    return f"/mnt/{drive}/{rest}"


def simplify(onnx_path: Path, out_path: Path, shapes: dict[str, list[int]]) -> bool:
    """用 onnxsim 折掉动态图元。返回是否真的写过文件。"""
    import onnx
    from onnxsim import simplify as onnxsim_simplify

    model = onnx.load(str(onnx_path))
    simplified, ok = onnxsim_simplify(
        model,
        overwrite_input_shapes=shapes,
        skip_fuse_bn=True,
    )
    if not ok:
        raise SystemExit("onnxsim 简化失败（模型可能有 onnxsim 不支持的算子）")
    onnx.save(simplified, str(out_path))
    return True


def converter_command(
    root: Path,
    platform: str,
    onnx_path: Path,
    out_cpp: Path,
    inputs: list[tuple[str, str]],
    act_bw: int,
    weight_bw: int,
    python: str,
    use_wsl: bool,
    distro: str | None,
) -> list[str]:
    """拼出转换器命令；Windows 上包一层 ``wsl -d <distro> -- bash -lc``。"""
    converter = find_converter(root, platform)
    lib_dir = root / "lib" / platform
    py_lib = root / "lib" / "python"

    if use_wsl:
        c = to_wsl_path(converter)
        p = to_wsl_path(py_lib)
        l = to_wsl_path(lib_dir)
        net = to_wsl_path(onnx_path)
        out = to_wsl_path(out_cpp)
        conv = [f"{python} {c}", "--input_network", net, "--output_path", out]
        conv += ["--act_bw", str(act_bw), "--weight_bw", str(weight_bw)]
        for name, dims in inputs:
            conv += ["-d", name, dims]
        inner = " && ".join(
            [
                f"export PYTHONPATH={p}",
                f"export LD_LIBRARY_PATH={l}",
                " ".join(conv),
            ]
        )
        cmd = ["wsl"]
        if distro:
            cmd += ["-d", distro]
        cmd += ["--", "bash", "-lc", inner]
        return cmd

    cmd = [
        python,
        str(converter),
        "--input_network",
        str(onnx_path),
        "--output_path",
        str(out_cpp),
        "--act_bw",
        str(act_bw),
        "--weight_bw",
        str(weight_bw),
    ]
    for name, dims in inputs:
        cmd += ["-d", name, dims]
    return cmd


def _search_upwards(root: Path, tail: str, marker: str) -> Path | None:
    """从 root 往上几层找 ``…/<tail>``（tail 形如 ``include/QNN``），marker 是里面必须有的文件。"""
    base = root
    for _ in range(5):
        direct = base / tail
        if (direct / marker).exists():
            return direct.resolve()
        for depth in (1, 2, 3):
            pattern = "/".join(["*"] * depth) + f"/{tail}/{marker}"
            for hit in base.glob(pattern):
                return hit.parent.resolve()
        base = base.parent
    return None


def find_jni(root: Path, explicit: str | None) -> Path:
    """找 QAIRT 的 ``share/QNN/converter/jni``（含 QnnModel 包装层源码）。"""
    if explicit:
        cand = Path(explicit)
        if (cand / "QnnModel.cpp").exists():
            return cand.resolve()
        raise SystemExit(f"--jni 里没有 QnnModel.cpp：{cand}")
    found = _search_upwards(root, "share/QNN/converter/jni", "QnnModel.cpp")
    if found:
        return found
    raise SystemExit(
        "找不到 QAIRT 的 jni 支撑源码（QnnModel.cpp / QnnWrapperUtils.cpp）。\n"
        "用 --jni 指定，或从 SDK zip 里解出 share/QNN/converter/jni。"
    )


def find_include(root: Path, explicit: str | None) -> Path:
    """找 QNN 头文件目录（含 ``QnnInterface.h``）。"""
    if explicit:
        cand = Path(explicit)
        if (cand / "QnnInterface.h").exists():
            return cand.resolve()
        raise SystemExit(f"--include 里没有 QnnInterface.h：{cand}")
    found = _search_upwards(root, "include/QNN", "QnnInterface.h")
    if found:
        return found
    raise SystemExit(
        "找不到 QNN 头文件目录（需要 QnnInterface.h）。\n"
        "用 --include 指定，或从 SDK zip 里解出 include/QNN。"
    )


RUN_SH = """#!/bin/sh
# 在 Android 设备上跑：sh run.sh
# 依赖本目录里的 QNN 运行库（LD_LIBRARY_PATH/ADSP_LIBRARY_PATH 都指向 .）
set -e
export LD_LIBRARY_PATH=.
export ADSP_LIBRARY_PATH=.

./qnn-context-binary-generator --backend libQnnHtp.so --model lib%(name)s.so \\
    --output_dir . --binary_file %(name)s_ctx.bin

# 这个版本的 qnn-net-run 没有 --perf_wall_clock / --perf_profile
# （会报 "Unused Arguments" 并退出），计时靠 --profiling_level 生成的 log。
start=$(date +%%s%%N)
./qnn-net-run --backend libQnnHtp.so --retrieve_context %(name)s_ctx.bin \\
    --input_list input_list.txt --output_dir out --num_inferences 20 --profiling_level=basic
end=$(date +%%s%%N)
echo "wall clock for 20 inferences: $(( (end - start) / 1000000 )) ms"
ls -la out || true
"""


def stage_bundle(
    name: str,
    out_dir: Path,
    stage: Path,
    root: Path,
    shapes: dict[str, list[int]],
) -> None:
    """把模型库 + QNN 运行库 + 随机输入 + run.sh 拼成可直接 adb push 的目录。"""
    stage.mkdir(parents=True, exist_ok=True)

    wanted: list[tuple[Path, str]] = [(out_dir / f"{name}.bin", f"{name}.bin")]
    so = out_dir / f"lib{name}.so"
    if so.exists():
        wanted.append((so, so.name))

    # SDK 常被拆成多份解包（bin+lib 一份、share 一份、include 一份），
    # 所以运行库也要往上找。
    dev_bin = _search_upwards(root, "bin/aarch64-android", "qnn-net-run")
    dev_lib = _search_upwards(root, "lib/aarch64-android", "libQnnHtp.so")
    if dev_bin is None or dev_lib is None:
        raise SystemExit(
            "找不到 aarch64-android 的 QNN 运行库（bin/aarch64-android + lib/aarch64-android），"
            "--stage 需要它们。检查 SDK 解包是否完整。"
        )
    for src, dst in (
        (dev_bin / "qnn-net-run", "qnn-net-run"),
        (dev_bin / "qnn-context-binary-generator", "qnn-context-binary-generator"),
        (dev_lib / "libQnnHtp.so", "libQnnHtp.so"),
        (dev_lib / "libQnnHtpPrepare.so", "libQnnHtpPrepare.so"),
        (dev_lib / "libQnnHtpV81Stub.so", "libQnnHtpV81Stub.so"),
        (dev_lib / "libQnnSystem.so", "libQnnSystem.so"),
    ):
        if src.exists():
            wanted.append((src, dst))
    for base in (dev_lib, dev_lib.parent):
        for hit in sorted(base.glob("hexagon-*/unsigned/libQnnHtpV*Skel.so")):
            wanted.append((hit, hit.name))

    for src, dst in wanted:
        shutil.copy2(src, stage / dst)

    # 随机输入：按 --input 的维度生成 raw，并写 input_list.txt
    lines: list[str] = []
    import numpy as np

    for input_name, dims in shapes.items():
        count = 1
        for d in dims:
            count *= d
        # QNN 侧模型库把 codes 这类整型张量标成 INT32；这里统一按 int32 造，
        # 浮点输入按 float32 造。真实数据替换同名 .raw 即可。
        if input_name in {"codes", "text"}:
            array = np.random.randint(0, 1000, size=count, dtype=np.int32)
        else:
            array = np.random.rand(count).astype(np.float32)
        path = stage / f"{input_name}.raw"
        array.tofile(path)
        lines.append(f"./{path.name}")
    # qnn-net-run 的 input_list 是"一行一次推理、同一行内用空格分隔各个输入"；
    # 每个输入单独一行的写法会被理解成"只有 1 个输入"，直接报错。
    (stage / "input_list.txt").write_text(" ".join(lines) + "\n", encoding="utf-8")

    (stage / "run.sh").write_text(RUN_SH % {"name": name}, encoding="utf-8")
    print(f"[stage] {stage}（{len(wanted)} 个库/权重 + {len(lines)} 个随机输入 + run.sh）")
    print(f"[stage] 用法：adb push \"{stage}\" /data/local/tmp/{name} && adb shell 'cd /data/local/tmp/{name} && chmod 755 qnn* && sh run.sh'")


def find_ndk(explicit: str | None) -> Path:
    """找 Android NDK 根目录（含 ``toolchains/llvm``）。"""
    def ok(p: Path) -> bool:
        return (p / "toolchains" / "llvm").is_dir()

    if explicit:
        cand = Path(explicit)
        if ok(cand):
            return cand.resolve()
        raise SystemExit(f"--ndk 不像 NDK 根目录（缺 toolchains/llvm）：{cand}")

    for env in ("ANDROID_NDK_HOME", "ANDROID_NDK_ROOT"):
        value = os.environ.get(env)
        if value and ok(Path(value)):
            return Path(value).resolve()

    bases: list[Path] = []
    for env in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
        value = os.environ.get(env)
        if value:
            bases.append(Path(value) / "ndk")
    home = Path.home()
    bases += [
        Path(os.environ.get("LOCALAPPDATA", home / "AppData/Local")) / "Android/Sdk/ndk",
        Path("D:/Android/Sdk/ndk"),
        Path("C:/Android/Sdk/ndk"),
    ]
    hits: list[Path] = []
    for base in bases:
        if base.is_dir():
            hits += [p for p in base.iterdir() if p.is_dir() and ok(p)]

    def version_key(p: Path) -> tuple[int, ...]:
        return tuple(int(x) for x in p.name.split(".") if x.isdigit())

    if hits:
        return sorted(hits, key=version_key)[-1].resolve()
    raise SystemExit(
        "--build-so 需要 NDK：用 --ndk 指定，或设 ANDROID_NDK_HOME / ANDROID_HOME。\n"
        "SDK 里装一个 NDK：sdkmanager \"ndk;27.3.13750724\""
    )


def build_model_lib(
    name: str,
    cpp: Path,
    out_so: Path,
    sdk_root: Path,
    ndk: Path,
    api: int,
    jni: Path,
    include: Path,
    weights_bin: Path | None = None,
) -> list[str]:
    """用 NDK 把转换出来的源码编成 Android 模型库。"""
    needed = [
        jni / "QnnModel.cpp",
        jni / "QnnWrapperUtils.cpp",
        jni / "linux" / "QnnModelPal.cpp",
    ]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        raise SystemExit(
            "缺少 QAIRT 的 jni 支撑源码，先从 SDK zip 里解出 share/QNN/converter/jni：\n  "
            + "\n  ".join(missing)
        )

    clang = ndk / "toolchains" / "llvm" / "prebuilt"
    host = "windows-x86_64" if os.name == "nt" else "linux-x86_64"
    cxx = clang / host / "bin" / ("clang++.exe" if os.name == "nt" else "clang++")
    if not cxx.exists():
        raise SystemExit(f"找不到 NDK clang++：{cxx}")

    objects = weight_objects(name, weights_bin, out_so.parent, ndk, api)
    cmd = [
        str(cxx),
        "-shared",
        "-fPIC",
        "-O2",
        "-std=c++17",
        f"--target=aarch64-linux-android{api}",
        "-Wno-c99-designator",
        f"-I{include}",
        f"-I{jni}",
        f"-I{jni / 'linux'}",
        str(cpp),
        str(jni / "QnnModel.cpp"),
        str(jni / "QnnWrapperUtils.cpp"),
        str(jni / "linux" / "QnnModelPal.cpp"),
        "-o",
        str(out_so),
        "-llog",
        "-ldl",
    ]
    if objects is not None:
        cmd.insert(-4, "@" + str(objects))
    return cmd


def weight_objects(
    name: str,
    weights_bin: Path | None,
    work: Path,
    ndk: Path,
    api: int,
) -> Path | None:
    """把 <name>.bin 里的 .raw 张量变成可链接的 .o 列表。

    转换器产出的 `<name>.bin` 其实是个 tar 包，里面是每个权重张量一个 `.raw`；
    `.cpp` 里靠 `BINVARSTART(...)` 引用 `_binary_obj_binary_<n>_raw_start` 符号，
    所以模型库必须把这些 `.raw` 各自 `objcopy -I binary` 成 `.o` 再一起链接，
    否则 dlopen 时会报 "cannot locate symbol _binary_obj_binary_..._raw_start"。
    """
    if weights_bin is None or not weights_bin.exists():
        return None
    if not tarfile.is_tarfile(weights_bin):
        print(
            f"[weight-objects] {weights_bin.name} 不是 tar 包，跳过权重对象"
            "（只有 contex-binary 版 .bin 才需要这一步）"
        )
        return None

    clang = ndk / "toolchains" / "llvm" / "prebuilt"
    host = "windows-x86_64" if os.name == "nt" else "linux-x86_64"
    objcopy = clang / host / "bin" / ("llvm-objcopy.exe" if os.name == "nt" else "llvm-objcopy")
    if not objcopy.exists():
        raise SystemExit(f"找不到 llvm-objcopy：{objcopy}")

    # The symbol name llvm-objcopy emits comes from the path handed to it, and
    # the generated .cpp asks for `_binary_obj_binary_<n>_raw_start`, so the
    # extraction root has to be literally `obj/binary` below the working dir.
    raw_dir = work / "obj" / "binary"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(weights_bin) as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(".raw")]
        tar.extractall(raw_dir, members=members)
    print(f"[weight-objects] {len(members)} 个权重张量，解到 {raw_dir}")

    objs: list[Path] = []
    for member in members:
        raw = raw_dir / member.name
        obj = raw.with_suffix(".o")
        rel_raw = f"obj/binary/{member.name}"
        rel_obj = f"obj/binary/{Path(member.name).with_suffix('.o').name}"
        subprocess.run(
            [
                str(objcopy),
                "--input-target=binary",
                "--output-target=elf64-littleaarch64",
                "--binary-architecture=aarch64",
                rel_raw,
                rel_obj,
            ],
            check=True,
            cwd=str(work),
        )
        objs.append(obj)
    print(f"[weight-objects] objcopy 完成 {len(objs)} 个 .o")

    rsp = work / f"{name}-objects.rsp"
    # clang parses @<file> lists itself, where backslashes escape, so Windows
    # paths have to be written with forward slashes.
    rsp.write_text("\n".join(f'"{o.as_posix()}"' for o in objs), encoding="utf-8")
    return rsp


def run(cmd: list[str], label: str) -> None:
    print(f"[{label}] {' '.join(cmd)}")
    started = time.time()
    proc = subprocess.run(cmd)
    elapsed = time.time() - started
    if proc.returncode != 0:
        raise SystemExit(f"{label} 失败（exit {proc.returncode}，{elapsed:.1f}s）")
    print(f"[{label}] ok {elapsed:.1f}s")


def pick_wsl_distro(python: str) -> str | None:
    """挑一个有可用解释器（python3.10 + numpy<2 + onnx）的 WSL 发行版。

    默认发行版不一定是装了 venv 的那个（本机默认 Ubuntu-24.04 是 3.12），
    所以逐个试一遍：能 import numpy/onnx 且 numpy 是 1.x 才算数。
    """
    listing = subprocess.run(
        ["wsl", "-l", "-q"], capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if listing.returncode != 0:
        return None
    probe = f'{python} -c "import numpy, onnx; assert numpy.__version__.startswith(\'1.\')"'
    for raw in listing.stdout.replace("\x00", "").splitlines():
        distro = raw.strip()
        if not distro:
            continue
        res = subprocess.run(
            ["wsl", "-d", distro, "--", "bash", "-lc", probe],
            capture_output=True,
        )
        if res.returncode == 0:
            return distro
    return None


def main() -> int:
    # Windows 控制台默认 GBK，中文提示会变乱码；统一按 UTF-8 输出。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="ONNX → QNN(HTP) 模型，模型无关的最小流水线",
    )
    parser.add_argument("--onnx", required=True, help="输入 ONNX（动态轴最好已固定）")
    parser.add_argument("--out-dir", required=True, help="产物目录")
    parser.add_argument("--name", default=None, help="模型名，默认取 onnx 文件名")
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="NAME=DIMS",
        help="固定输入形状，例如 codes=1,1,64；可重复",
    )
    parser.add_argument("--act-bw", type=int, default=16, help="激活位宽（8/16）")
    parser.add_argument("--weight-bw", type=int, default=16, help="权重位宽（8/16）")
    parser.add_argument("--sdk-root", default=None, help="QAIRT 根目录（默认读 QNN_SDK_ROOT）")
    parser.add_argument("--platform", default=None, choices=PLATFORMS)
    parser.add_argument("--python", default=None, help="跑转换器的解释器（需 3.10 + numpy<2）")
    parser.add_argument(
        "--no-wsl",
        action="store_true",
        help="Windows 上也直接调用 converter（默认自动进 WSL）",
    )
    parser.add_argument("--distro", default=None, help="WSL 发行版名，例如 Ubuntu-22.04")
    parser.add_argument("--no-simplify", action="store_true", help="跳过 onnxsim")
    parser.add_argument("--build-so", action="store_true", help="顺带编出 Android 模型库")
    parser.add_argument("--ndk", default=None, help="NDK 根目录（--build-so 时需要）")
    parser.add_argument(
        "--jni",
        default=None,
        help="QAIRT 的 share/QNN/converter/jni 目录（--build-so 时需要，默认自动找）",
    )
    parser.add_argument(
        "--stage",
        default=None,
        help="同时拼出一个可直接 adb push 的目录（模型库 + QNN 运行库 + 随机输入 + run.sh）",
    )
    parser.add_argument(
        "--include",
        default=None,
        help="QNN 头文件目录（--build-so 时需要，默认自动找）",
    )
    parser.add_argument("--api", type=int, default=26, help="--build-so 的 Android API level")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令")
    args = parser.parse_args()

    onnx_path = Path(args.onnx).resolve()
    if not onnx_path.exists():
        raise SystemExit(f"ONNX 不存在：{onnx_path}")
    name = args.name or onnx_path.stem
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = parse_inputs(args.input)
    shapes = {n: [int(d) for d in dims.split(",")] for n, dims in inputs}

    root = find_sdk_root(args.sdk_root)
    use_wsl = os.name == "nt" and not args.no_wsl
    # WSL 里跑的是 Linux 版 converter，别拿 Windows 版（那版在 numpy 2.x 下会崩）。
    platform = args.platform or (
        "x86_64-linux-clang"
        if use_wsl or os.name != "nt"
        else "x86_64-windows-msvc"
    )
    python = args.python or ("~/qvenv/bin/python" if use_wsl else sys.executable)
    distro = args.distro
    if use_wsl and not distro:
        distro = os.environ.get("AHA_WSL_DISTRO") or pick_wsl_distro(python)
        if not distro:
            raise SystemExit(
                "没找到装了 python3.10 + numpy<2 + onnx 的 WSL 发行版。\n"
                "先在里面建个 venv（转换器要 cp310）：\n"
                "  wsl -d Ubuntu-22.04 -- bash -lc 'python3 -m venv ~/qvenv && "
                "~/qvenv/bin/pip install -i https://mirrors.aliyun.com/pypi/simple "
                "onnx==1.17.0 \"numpy<2\" pyyaml packaging jsonschema pandas'\n"
                "然后重跑，或用 --distro / --python 显式指定。"
            )

    # 1) onnxsim
    net = onnx_path
    if not args.no_simplify:
        net = out_dir / f"{name}.sim.onnx"
        if args.dry_run:
            print(f"[simplify] {onnx_path} -> {net}（{shapes}）")
        else:
            started = time.time()
            simplify(onnx_path, net, shapes)
            print(f"[simplify] ok {time.time() - started:.1f}s -> {net}")

    cpp = out_dir / f"{name}.cpp"
    weights_bin = out_dir / f"{name}.bin"
    # 2) converter
    cmd = converter_command(
        root,
        platform,
        net,
        cpp,
        inputs,
        args.act_bw,
        args.weight_bw,
        python,
        use_wsl,
        distro,
    )
    if args.dry_run:
        print(f"[convert] {' '.join(cmd)}")
    else:
        run(cmd, "convert")

    # 3) 可选：模型库
    if args.build_so:
        ndk = find_ndk(args.ndk)
        so = out_dir / f"lib{name}.so"
        jni = find_jni(root, args.jni)
        include = find_include(root, args.include)
        cmd = build_model_lib(name, cpp, so, root, ndk, args.api, jni, include, weights_bin)
        if args.dry_run:
            print(f"[build-so] {' '.join(cmd)}")
        else:
            run(cmd, "build-so")

    if args.stage and not args.dry_run:
        stage_bundle(name, out_dir, Path(args.stage), root, shapes)

    meta = {
        "onnx": str(onnx_path),
        "simplified": str(net) if net != onnx_path else None,
        "name": name,
        "inputs": [{"name": n, "dims": shapes[n]} for n, _ in inputs],
        "act_bitwidth": args.act_bw,
        "weights_bitwidth": args.weight_bw,
        "sdk_root": str(root),
        "platform": platform,
        "via_wsl": use_wsl,
    }
    if not args.dry_run:
        meta_path = out_dir / f".{name}.meta.json"
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[meta] {meta_path}")
        for produced in (cpp, out_dir / f"{name}.bin"):
            if produced.exists():
                print(f"  {produced.name}: {produced.stat().st_size:,} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




