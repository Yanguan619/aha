#!/usr/bin/env python3
"""GPT-SoVITS 中文前端的参考实现（用于给 Rust 移植做对拍）。

复用仓库里的真实代码（text/chinese2.py、text/tone_sandhi.py、zh_normalization），
只把两处替换掉：

1. ``jieba_fast`` -> ``jieba``（API 兼容；jieba_fast 只是 C 加速版，词典相同）
2. ``text.g2pw`` -> 一个假的模块，多音字退回 pypinyin 默认读音

这样不需要 onnxruntime / transformers / torch 就能跑出 ``phones`` 和 ``word2ph``，
是 Rust 侧 P1 的黄金基准。注意：真实 GPT-SoVITS 开着 G2PW，所以**多音字**的差异是
预期的，其余必须逐元素一致。

用法::

    python reference_frontend.py --gsv-root D:/GPT-SoVITS "今天天气不错，我们出去走走吧。"
    python reference_frontend.py --gsv-root D:/GPT-SoVITS --jsonl corpus.txt --out ref.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path


def install_stubs(gsv_root: Path) -> None:
    import jieba
    import jieba.posseg

    # 1) jieba_fast -> jieba
    jieba.setLogLevel(20)
    sys.modules["jieba_fast"] = jieba
    sys.modules["jieba_fast.posseg"] = jieba.posseg

    # 2) text.g2pw -> pypinyin 兜底
    fake = types.ModuleType("text.g2pw")

    class _FakeG2PW:
        def __init__(self, *args, **kwargs):
            from pypinyin import Style, lazy_pinyin

            self._lazy = lazy_pinyin
            self._style = Style

        def _g2pw(self, segments):
            return [
                self._lazy(
                    seg,
                    style=self._style.TONE3,
                    neutral_tone_with_five=True,
                    v_to_u=False,
                )
                for seg in segments
            ]

    fake.G2PWPinyin = _FakeG2PW
    fake.correct_pronunciation = lambda word, pinyins: pinyins
    sys.modules["text.g2pw"] = fake

    sys.path.insert(0, str(gsv_root / "GPT_SoVITS"))


def main() -> int:
    ap = argparse.ArgumentParser(description="GPT-SoVITS 中文前端参考实现")
    ap.add_argument("--gsv-root", type=Path, required=True)
    ap.add_argument("text", nargs="*", help="要分析的中文句子")
    ap.add_argument("--jsonl", type=Path, help="批量输入：每行一句")
    ap.add_argument("--out", type=Path, help="批量输出 JSONL")
    args = ap.parse_args()

    install_stubs(args.gsv_root)

    from text import chinese2, symbols2
    from text import cleaned_text_to_sequence

    def analyze(text: str) -> dict:
        norm = chinese2.text_normalize(text)
        phones, word2ph = chinese2.g2p(norm)
        ids = cleaned_text_to_sequence(phones, "v2")
        return {
            "text": text,
            "norm_text": norm,
            "phones": phones,
            "ids": ids,
            "word2ph": word2ph,
            "num_symbols": len(symbols2.symbols),
        }

    if args.jsonl:
        lines = [ln.strip() for ln in args.jsonl.read_text(encoding="utf-8").splitlines()]
        records = [analyze(ln) for ln in lines if ln]
        out = args.out or Path("frontend_ref.jsonl")
        out.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
            encoding="utf-8",
        )
        print(f"{len(records)} 条 -> {out}")
        return 0

    for text in args.text or ["今天天气不错，我们出去走走吧。"]:
        rec = analyze(text)
        print(json.dumps(rec, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
