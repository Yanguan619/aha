#!/usr/bin/env python3
"""本地 CI：把「提交前该跑的东西」收成一条命令。

    python scripts/ci.py            # 主入口：clippy + 多 feature 编译 + 测试
    python scripts/ci.py fmt        # 就地格式化
    python scripts/ci.py fmt --check
    python scripts/ci.py clippy
    python scripts/ci.py check
    python scripts/ci.py test

只用标准库，不需要装任何依赖。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# 仓库根目录（本文件在 scripts/ 下）。
ROOT = Path(__file__).resolve().parent.parent

# 需要逐个 `cargo check` 的 feature 组合。
#
# - "default"：桌宠以外的通用能力，原来的 CI 只测这个；
# - 单个 feature：早一点定位是哪个后端编不过，比混在一起好查；
# - 全组合：桌宠真实构建的那一套（不含 cuda，那个要 nvcc）。
FEATURE_SETS: list[tuple[str, list[str]]] = [
    ("default", []),
    ("sherpa", ["sherpa"]),
    ("gpt-sovits", ["gpt-sovits"]),
    ("laya", ["laya"]),
    ("vla", ["vla"]),
    ("deskpet (all, no cuda)", ["sherpa", "gpt-sovits", "laya", "vla"]),
]

# 需要跑 `--lib` 测试的 feature 组合。
LIB_TEST_SETS: list[list[str]] = [
    [],
    ["gpt-sovits"],
    ["sherpa", "gpt-sovits", "laya", "vla"],
]

# 需要跑 `--bins` 测试的 feature 组合（服务端测试在 bin 里）。
BIN_TEST_SETS: list[list[str]] = [
    ["sherpa", "gpt-sovits", "laya", "vla"],
]


def run(step: str, args: list[str], failed: list[str]) -> bool:
    """跑一条命令；失败记进 `failed`，返回是否成功。"""
    print(f"\n==> {step}")
    print("    " + " ".join(args))
    try:
        code = subprocess.call(args, cwd=ROOT)
    except FileNotFoundError:
        print(f"    找不到命令：{args[0]}")
        failed.append(step)
        return False
    except KeyboardInterrupt:
        print("\n    被中断")
        raise
    if code != 0:
        failed.append(step)
        return False
    return True


def cargo(*args: str) -> list[str]:
    return ["cargo", *args]


def feature_args(features: list[str]) -> list[str]:
    return ["--features", ",".join(features)] if features else []


def step_fmt(failed: list[str], check: bool) -> None:
    args = ["fmt", "--all"]
    if check:
        args += ["--", "--check"]
    ok = run("cargo fmt", cargo(*args), failed)
    if not ok and check:
        print(
            "    提示：这个仓库是用旧版 rustfmt 格式化的，新版会想改一大批文件。\n"
            "          要统一刷一遍用 `python scripts/ci.py fmt`（会有较大 diff）。"
        )


def step_clippy(failed: list[str], deny: bool) -> None:
    """只在 `--lib --bins` 上 lint。

    不用 `--all-targets`：`tests/` 和 `examples/` 里有一批**依赖 feature 的**
    target（`test_gpt_sovits_*` 要 `gpt-sovits`、`test_cuda_graph` 要 `cuda`、
    `stage_probe` 要 `vla`），但它们没在 Cargo.toml 里声明 `required-features`，
    所以 `--all-targets` 在任何单一 feature 组合下都会因为解析不到模块而失败。
    （`make lint` 用的就是 `--all-targets`，因此它现在是红的。）
    """
    sets = [
        ("default", []),
        ("deskpet (all, no cuda)", ["sherpa", "gpt-sovits", "laya", "vla"]),
    ]
    for name, features in sets:
        args = ["clippy", "--no-deps", "--lib", "--bins", *feature_args(features)]
        if deny:
            args += ["--", "-D", "warnings"]
        ok = run(f"cargo clippy [{name}]", cargo(*args), failed)
        if not ok:
            print(
                "    提示：`sentencepiece-sys` 需要 cmake。Windows 上如果报\n"
                '          "failed to execute cmake"，装一个 cmake 并加进 PATH。'
            )
            return
    if not deny:
        print(
            "    提示：默认不把 warning 当失败——仓库里有一批既有 warning\n"
            "          （f32::NEG_INFINITY 在 Rust 1.99 被 deprecated、若干 unused）。\n"
            "          清理完之后加 --deny 收紧成 -D warnings。"
        )


def step_check(failed: list[str], with_cuda: bool) -> None:
    sets = list(FEATURE_SETS)
    if with_cuda:
        sets.append(("cuda + deskpet", ["cuda", "sherpa", "gpt-sovits", "laya", "vla"]))
    else:
        print("\n（跳过 cuda 组合；需要时加 --cuda，会用到 nvcc 和 MSVC 的 cl.exe）")

    for name, features in sets:
        run(
            f"check [{name}]",
            cargo("check", "--lib", "--bins", *feature_args(features)),
            failed,
        )


def step_test(failed: list[str]) -> None:
    for features in LIB_TEST_SETS:
        name = ",".join(features) if features else "default"
        run(
            f"test --lib [{name}]",
            cargo("test", "--lib", *feature_args(features)),
            failed,
        )
    for features in BIN_TEST_SETS:
        name = ",".join(features)
        run(
            f"test --bins [{name}]",
            cargo("test", "--bins", *feature_args(features)),
            failed,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="ci.py",
        description="本地 CI：clippy + 多 feature 编译 + 测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "例子：\n"
            "  python scripts/ci.py              # 本地 CI 主入口\n"
            "  python scripts/ci.py fmt          # 就地格式化\n"
            "  python scripts/ci.py check --cuda # 额外验证 cuda 组合\n"
            "  python scripts/ci.py clippy --deny  # 把 warning 当失败\n"
        ),
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="ci",
        choices=["ci", "fmt", "clippy", "check", "test"],
        help="要跑的任务（默认 ci）",
    )
    parser.add_argument("--check", action="store_true", help="fmt：只检查不修改")
    parser.add_argument("--cuda", action="store_true", help="check：额外跑 cuda 组合")
    parser.add_argument(
        "--deny", action="store_true", help="clippy：把 warning 当失败（-D warnings）"
    )
    args = parser.parse_args()

    if shutil.which("cargo") is None:
        print("找不到 cargo：请先装 Rust 工具链（https://rustup.rs）")
        return 2

    failed: list[str] = []
    if args.command == "ci":
        step_clippy(failed, args.deny)
        step_check(failed, args.cuda)
        step_test(failed)
    elif args.command == "fmt":
        step_fmt(failed, args.check)
    elif args.command == "clippy":
        step_clippy(failed, args.deny)
    elif args.command == "check":
        step_check(failed, args.cuda)
    elif args.command == "test":
        step_test(failed)

    if not failed:
        print("\n全部通过。")
        return 0
    print(f"\n失败 {len(failed)} 项：")
    for item in failed:
        print(f"  - {item}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
