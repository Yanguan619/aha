#!/usr/bin/env python3
"""导出 GPT-SoVITS 中文文本前端所需的数据包（不依赖 torch / 不依赖 GPT-SoVITS 运行环境）。

产出（放到模型目录里，Rust 侧启动时加载一次）：

    frontend_zh.json   char→拼音、词→拼音、opencpop 音素映射、变调词表、儿化词表、标点
    dict.txt           jieba 主词典（词 频次 词性），用于分词 + 词性查找

数据来源：
    pypinyin            拼音字典 / 词组字典（与 GPT-SoVITS 用的同一套）
    opencpop-strict.txt 拼音 → (声母, 韵母) 音素映射
    text/tone_sandhi.py 必读轻声词表 / 不得轻声词表
    text/chinese2.py    标点替换表、儿化词表（用 ast 静态抽取，不执行）

用法::

    python export_frontend_data.py --gsv-root D:/GPT-SoVITS --out D:/model_download/gpt-sovits-v2pro-zh

只依赖 pypinyin + jieba（用于定位 dict.txt）。
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import sys
import types
from pathlib import Path

from pypinyin import phrases_dict, pinyin_dict
from pypinyin.contrib.tone_convert import to_tone3


def _tone3(marked: str) -> str:
    """带调符号的拼音 -> tone3（例：lǎng -> lang3，儿化保留 v 写法）。"""
    return to_tone3(marked.strip(), v_to_u=False, neutral_tone_with_five=True)


def export_pinyin() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    chars: dict[str, list[str]] = {}
    for codepoint, variants in pinyin_dict.pinyin_dict.items():
        try:
            ch = chr(int(codepoint))
        except (ValueError, OverflowError):
            continue
        py = [_tone3(v) for v in str(variants).split(",") if v.strip()]
        if py:
            chars[ch] = py

    phrases: dict[str, list[str]] = {}
    for word, syllables in phrases_dict.phrases_dict.items():
        try:
            py = [_tone3(item[0] if isinstance(item, (list, tuple)) else item) for item in syllables]
        except Exception:
            continue
        if py and all(py):
            phrases[word] = py
    return chars, phrases


def export_syllables(chars: dict[str, list[str]], phrases: dict[str, list[str]]) -> dict[str, list[str]]:
    """每个带调音节 -> (声母, 韵母_tone3)。

    pypinyin 的切分对零声母有特殊规则（yin2 -> ("", "in2")，wo3 -> ("w", "o3")），
    直接在这里算好，Rust 侧只查表，避免逆向音系规则。
    """
    from pypinyin.contrib.tone_convert import to_finals_tone3, to_initials

    seen: set[str] = set()
    for variants in chars.values():
        seen.update(variants)
    for syllables in phrases.values():
        seen.update(syllables)

    table: dict[str, list[str]] = {}
    for syl in seen:
        if not syl or not syl[0].isalpha():
            continue
        try:
            initial = to_initials(syl)
            final = to_finals_tone3(syl, v_to_u=False, neutral_tone_with_five=True)
        except Exception:
            continue
        if final:
            table[syl] = [initial, final]
    return table


def export_opencpop(gsv_root: Path) -> dict[str, list[str]]:
    path = gsv_root / "GPT_SoVITS" / "text" / "opencpop-strict.txt"
    mapping: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        key, value = line.split("\t")
        mapping[key.strip()] = value.strip().split(" ")
    return mapping


def export_tone_sandhi(gsv_root: Path) -> dict[str, list[str]]:
    """静态执行 tone_sandhi.py 里的两个词表（jieba_fast 用空壳顶替）。"""
    stub = types.ModuleType("jieba_fast")
    stub.cut_for_search = lambda text: [text]
    stub.setLogLevel = lambda *a, **k: None
    sys.modules.setdefault("jieba_fast", stub)

    path = gsv_root / "GPT_SoVITS" / "text" / "tone_sandhi.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location("gsv_tone_sandhi", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    sandhi = mod.ToneSandhi()
    return {
        "must_neural": sorted(sandhi.must_neural_tone_words),
        "must_not_neural": sorted(sandhi.must_not_neural_tone_words),
        "punc": sorted(sandhi.punc),
    }


def export_chinese_constants(gsv_root: Path) -> dict:
    """用 ast 抽取 chinese2.py 的模块级常量（不执行、不引入 g2pw 依赖）。"""
    path = gsv_root / "GPT_SoVITS" / "text" / "chinese2.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    wanted = {"rep_map", "must_erhua", "not_erhua"}
    out: dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    try:
                        out[target.id] = ast.literal_eval(node.value)
                    except ValueError:
                        pass
    return out


def export_punctuation(gsv_root: Path) -> list[str]:
    path = gsv_root / "GPT_SoVITS" / "text" / "symbols.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    punctuation = ["!", "?", "…", ",", ".", "-"]
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "punctuation" for t in node.targets
        ):
            try:
                punctuation = list(ast.literal_eval(node.value))
            except ValueError:
                pass
            break
    if "-" not in punctuation:
        punctuation.append("-")
    return punctuation


def find_jieba_dict() -> Path:
    import jieba

    path = Path(jieba.__file__).parent / "dict.txt"
    if not path.is_file():
        raise SystemExit("找不到 jieba 的 dict.txt")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="导出 GPT-SoVITS 中文前端数据包")
    ap.add_argument("--gsv-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    chars, phrases = export_pinyin()
    syllables = export_syllables(chars, phrases)
    opencpop = export_opencpop(args.gsv_root)
    sandhi = export_tone_sandhi(args.gsv_root)
    consts = export_chinese_constants(args.gsv_root)

    pack = {
        "format": "aha-gpt-sovits-frontend-zh",
        "format_version": 1,
        "chars": chars,
        "phrases": phrases,
        "syllables": syllables,
        "opencpop": opencpop,
        "tone_sandhi": sandhi,
        "rep_map": consts.get("rep_map", {}),
        "must_erhua": sorted(consts.get("must_erhua", [])),
        "not_erhua": sorted(consts.get("not_erhua", [])),
        "punctuation": export_punctuation(args.gsv_root),
    }

    path = out / "frontend_zh.json"
    path.write_text(json.dumps(pack, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    dict_src = find_jieba_dict()
    shutil.copy2(dict_src, out / "dict.txt")

    print(f"chars         {len(chars)}")
    print(f"phrases       {len(phrases)}")
    print(f"syllables     {len(syllables)}")
    print(f"opencpop      {len(opencpop)}")
    print(f"must_neural   {len(sandhi['must_neural'])} / must_not {len(sandhi['must_not_neural'])}")
    print(f"erhua         {len(pack['must_erhua'])} / not {len(pack['not_erhua'])}")
    print(f"rep_map       {len(pack['rep_map'])}")
    print(f"-> {path}  {path.stat().st_size / 1024 / 1024:.1f} MiB")
    print(f"-> {out / 'dict.txt'}  {(out / 'dict.txt').stat().st_size / 1024 / 1024:.1f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
