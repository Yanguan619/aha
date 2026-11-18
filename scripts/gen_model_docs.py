#!/usr/bin/env python3
"""用 `aha list --json` 的输出重写 docs/supported-models{,.zh-CN}.md 里的模型清单块。

唯一来源是 `WhichModel` 枚举（`aha list` 从它导出），所以文档不再手抄：

    cargo run -q --features sherpa,gpt-sovits,laya,vla -- list --json > /tmp/models.json
    python scripts/gen_model_docs.py /tmp/models.json

加 `--check` 只校验不写盘（CI 里可以拿它当门禁）。
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = {
    "docs/supported-models.md": (
        "> Generated from `aha list --json` (the single source is the `WhichModel` enum) — **do not hand-edit**.\n"
        "> After adding a model, regenerate with `cargo run -- list --json` + `scripts/gen_model_docs.py`."
    ),
    "docs/supported-models.zh-CN.md": (
        "> 下表由 `aha list --json` 生成（唯一来源是 `WhichModel` 枚举），**不要手改**；\n"
        "> 新增模型后跑 `cargo run -- list --json` + `scripts/gen_model_docs.py` 重生成。"
    ),
}
FENCE = ("```shell", "```")


def render(models: list[dict]) -> list[str]:
    id_w = max(len("Model ID"), *(len(m["model_id"]) for m in models)) + 2
    owner_w = max(len("Owner"), *(len(m["owner"]) for m in models)) + 2
    out = [FENCE[0], "Available models:", ""]
    out.append(f"{'Model ID':<{id_w}}{'Owner':<{owner_w}}{'type'}")
    out.append("-" * (id_w + owner_w + 10))
    for m in models:
        out.append(f"{m['model_id']:<{id_w}}{m['owner']:<{owner_w}}{m['type']}")
    out.append(FENCE[1])
    return out


def rewrite(path: pathlib.Path, models: list[dict], note: str, check: bool) -> bool:
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith(FENCE[0]))
    end = next(i for i in range(start + 1, len(lines)) if lines[i].strip() == FENCE[1])

    # 表格前那两行说明（若存在）先摘掉，再统一重写
    head = list(lines[:start])
    # 去掉紧贴 fence 的说明块——可能有多段（中间隔空行），所以反复剥
    changed = True
    while changed:
        changed = False
        while head and head[-1].strip() == "":
            head.pop()
            changed = True
        while head and head[-1].startswith(">"):
            head.pop()
            changed = True

    body = head + ["", note, *render(models)] + lines[end + 1 :]
    new = "\n".join(body)
    if new == text:
        print(f"  {path.name}: 已是最新")
        return True
    if check:
        print(f"  {path.name}: 与枚举不一致（跑一次本脚本重生成）")
        return False
    path.write_text(new, encoding="utf-8")
    print(f"  {path.name}: 已重写（{len(models)} 行）")
    return True


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check = "--check" in sys.argv
    if not args:
        print(__doc__)
        return 2
    models = json.loads(pathlib.Path(args[0]).read_text(encoding="utf-8"))
    ok = True
    for rel, note in DOCS.items():
        ok &= rewrite(ROOT / rel, models, note, check)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
