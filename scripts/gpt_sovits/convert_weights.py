#!/usr/bin/env python3
"""GPT-SoVITS -> safetensors 转换脚本（给 aha / candle 用）。

GPT-SoVITS 官方权重是 PyTorch 的 .pth / .ckpt（zip + pickle），candle 读不了，而且
里面带 weight_norm（`*_g` / `*_v` 成对存放），需要在这里就地合并成普通卷积核。

**不依赖 torch**：用同目录的 `pth_reader.py` 直接解析 PyTorch 的 zip 格式。
（如果你的权重是 PyTorch < 1.6 的旧格式，加 `--backend torch`。）

用法：

    python convert_weights.py \
        --gsv-root D:/GPT-SoVITS \
        --out      D:/model_download/gpt-sovits-v2pro-zh \
        --version  v2Pro

    # 也可以逐个指定权重文件
    python convert_weights.py --out ./gsv \
        --gpt    D:/GPT-SoVITS/GPT_SoVITS/pretrained_models/s1v3.ckpt \
        --sovits D:/GPT-SoVITS/GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth \
        --sv     D:/GPT-SoVITS/GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt

产物（一个目录 = 一整套后备模型，可直接作为 aha 的 --weight-path）：

    manifest.json                 版本 / hps / t2s 配置 / 张量清单 / 参数统计
    gpt.safetensors               s1 Text2Semantic（AR）权重
    sovits.safetensors            s2G 合成器（已去 weight_norm）
    vocoder.safetensors           v3 / v4 专用声码器
    hubert.safetensors            chinese-hubert-base
    bert.safetensors              chinese-roberta-wwm-ext-large（去掉 MLM 头）
    sv_eres2net.safetensors       v2Pro / v2ProPlus 的说话人向量模型
    symbols.json                  phone 符号表（v2 系 732 个）
    hubert_config.json / bert_config.json / bert_tokenizer/*

依赖：numpy、safetensors。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pth_reader import load as load_pth  # noqa: E402

# ---------------------------------------------------------------------------
# 版本识别（与 GPT_SoVITS/process_ckpt.py 保持一致）
# ---------------------------------------------------------------------------

# 文件头前两字节被改成版本标记（process_ckpt.my_save2），读取时补回 "PK" 即可当 zip 解析。
HEAD_TO_VERSION = {
    b"00": ("v1", "v1", False),
    b"01": ("v2", "v2", False),
    b"02": ("v2", "v3", False),
    b"03": ("v2", "v3", True),
    b"04": ("v2", "v4", True),
    b"05": ("v2", "v2Pro", False),
    b"06": ("v2", "v2ProPlus", False),
    b"07": ("v2", "v5dev", True),
    b"08": ("v2", "v5turbo", True),
}

# 官方预训练底模：文件头是正常 PK，只能靠前 8KB 的 md5 区分。
HASH_TO_VERSION = {
    "dc3c97e17592963677a4a1681f30c653": ("v2", "v2", False),  # s2G488k.pth（v1 底模）
    "43797be674a37c1c83ee81081941ed0f": ("v2", "v3", False),  # s2Gv3.pth
    "6642b37f3dbb1f76882b69937c95a5f3": ("v2", "v2", False),  # s2G2333k.pth
    "4f26b9476d0c5033e04162c486074374": ("v2", "v4", False),  # s2Gv4.pth
    "c7e9fce2223f3db685cdfa1e6368728a": ("v2", "v2Pro", False),  # s2Gv2Pro.pth
    "66b313e39455b57ab1b0bc0b239c9d0a": ("v2", "v2ProPlus", False),  # s2Gv2ProPlus.pth
    "c4ce1f839a0d271e41963f9fb16ade76": ("v2", "v5dev", False),
    "6bc5cb2a195cc8ec22b2e723cd4d6f47": ("v2", "v5turbo", False),
}

V2PRO_VERSIONS = {"v2Pro", "v2ProPlus"}
V3V4_VERSIONS = {"v3", "v4", "v5dev", "v5turbo"}

# 推理只用得到的子模块；enc_q（posterior encoder）只服务训练。
KEEP_PREFIXES = (
    "enc_p.",
    "dec.",
    "flow.",
    "ref_enc.",
    "ssl_proj.",
    "quantizer.",
    "sv_emb.",
    "ge_to512.",
    "prelu.",
)

# 这些顶层键是训练残留，不是权重。
NON_WEIGHT_KEYS = {"scaler_state_dict", "optimizer", "lr_scheduler", "epoch", "global_step"}


def _md5_head(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read(8192)).hexdigest()


def detect_sovits_version(path: Path) -> tuple[str, str, bool]:
    """返回 (arch_version, model_version, is_lora)。"""
    with open(path, "rb") as f:
        head = f.read(2)
    if head in HEAD_TO_VERSION:
        return HEAD_TO_VERSION[head]
    hit = HASH_TO_VERSION.get(_md5_head(path))
    if hit is not None:
        return hit
    if head != b"PK":
        raise SystemExit(
            f"无法识别 {path.name} 的版本（文件头 {head!r}）。请用 --model-version 显式指定。"
        )
    size = path.stat().st_size
    if size < 82978 * 1024:
        return ("v1", "v1", False)
    if size < 700 * 1024 * 1024:
        return ("v2", "v2", False)
    return ("v2", "v3", False)


def replay_head_fix(path: Path, scratch: Path) -> Path:
    """把被改掉 "PK" 头的权重还原成可解析的临时文件。"""
    with open(path, "rb") as f:
        head = f.read(2)
    if head == b"PK":
        return path
    scratch.mkdir(parents=True, exist_ok=True)
    fixed = scratch / (path.name + ".fixed.pth")
    with open(fixed, "wb") as f:
        f.write(b"PK" + path.read_bytes()[2:])
    return fixed


# ---------------------------------------------------------------------------
# 权重整理
# ---------------------------------------------------------------------------


# weight_norm 的归一化维度因模块而异：
#   nn.utils.weight_norm(conv)            -> g 形状 (out,1,1)   ，按输出通道归一化
#   nn.utils.weight_norm(conv, dim=2)     -> g 形状 (1,1,k)     ，按 kernel 位置归一化（HuBERT 的位置卷积就是这种）
# 所以不能假定 dim=0，必须从 g 的形状反推。
DIM_STATS: dict[str, int] = {}


def _weight_norm_dims(g: np.ndarray, v: np.ndarray) -> tuple[int, ...]:
    """g 中大小为 1、而 v 中对应维大于 1 的维度，就是被归一化掉的维度。"""
    dims = tuple(
        i
        for i in range(v.ndim)
        if g.shape[i] == 1 and v.shape[i] > 1
    )
    if not dims:
        # g 与 v 同形（dim=None 的情况）：退化为按输出通道归一化
        dims = tuple(range(1, v.ndim))
    return dims


def merge_weight_norm(sd: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], int]:
    """把 weight_norm 的 (weight_g, weight_v) 合并成 weight。"""
    merged = 0
    out: dict[str, np.ndarray] = {}
    for key, arr in sd.items():
        if key.endswith(".weight_g"):
            continue
        if key.endswith(".weight_v"):
            base = key[: -len("weight_v")]
            g = sd[base + "weight_g"].astype(np.float64)
            v = arr.astype(np.float64)
            dims = _weight_norm_dims(g, v)
            shape = tuple(1 if i in dims else v.shape[i] for i in range(v.ndim))
            g = g.reshape(shape)
            norm = np.sqrt((v ** 2).sum(axis=dims, keepdims=True))
            out[base + "weight"] = (v * (g / norm)).astype(arr.dtype)
            key_dims = ",".join(str(d) for d in dims)
            DIM_STATS[key_dims] = DIM_STATS.get(key_dims, 0) + 1
            merged += 1
            continue
        out[key] = arr
    return out, merged


def report_modules(sd: dict[str, np.ndarray], label: str) -> dict[str, int]:
    stats: dict[str, int] = defaultdict(int)
    for key, arr in sd.items():
        stats[key.split(".")[0]] += int(arr.size)
    pretty = {k: round(v / 1e6, 2) for k, v in sorted(stats.items(), key=lambda kv: -kv[1])}
    print(f"    {label} 各模块参数(M): {pretty}")
    return pretty


def to_f32(sd: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    for key, arr in sd.items():
        if arr.dtype == np.float16:
            arr = arr.astype(np.float32)
        out[key] = np.ascontiguousarray(arr)
    return out


def save_safetensors(path: Path, sd: dict[str, np.ndarray]) -> dict[str, list[int]]:
    if not sd:
        raise SystemExit(f"{path.name}: 没有可保存的张量")
    save_file(sd, str(path))
    return {k: list(v.shape) for k, v in sd.items()}


def load_weights(path: Path, backend: str, scratch: Path) -> dict:
    if backend == "torch":
        import torch  # noqa: F401

        real = replay_head_fix(path, scratch)
        obj = torch.load(real, map_location="cpu", weights_only=False)
        conv = lambda v: v.float().numpy() if hasattr(v, "numpy") else v  # noqa: E731
        return {k: conv(v) if hasattr(v, "shape") else v for k, v in obj.items()}
    return load_pth(replay_head_fix(path, scratch))


# ---------------------------------------------------------------------------
# 来自 GPT-SoVITS 源码的辅助信息
# ---------------------------------------------------------------------------


def load_symbols(gsv_root: Path, arch_version: str) -> list[str] | None:
    """导出 phone 符号表（v1 用 text/symbols.py，v2 起用 text/symbols2.py）。"""
    module_name = "symbols" if arch_version == "v1" else "symbols2"
    path = gsv_root / "GPT_SoVITS" / "text" / f"{module_name}.py"
    if not path.is_file():
        return None
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"gsv_{module_name}", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return list(mod.symbols)


def copy_if_exists(src: Path, dst: Path) -> bool:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def convert_hubert(src_dir: Path, out_dir: Path, scratch: Path, backend: str):
    bin_path = src_dir / "pytorch_model.bin"
    if not bin_path.is_file():
        raise SystemExit(f"缺少 {bin_path}")
    sd = load_weights(bin_path, backend, scratch)
    sd.pop("masked_spec_embed", None)  # 仅训练用
    DIM_STATS.clear()
    sd, merged = merge_weight_norm(sd)
    print(f"[hubert] {len(sd)} 张量，合并 {merged} 组 weight_norm（归一化维度分布: {DIM_STATS}）")
    report_modules(sd, "hubert")
    shapes = save_safetensors(out_dir / "hubert.safetensors", to_f32(sd))
    copy_if_exists(src_dir / "config.json", out_dir / "hubert_config.json")
    return shapes


def convert_bert(src_dir: Path, out_dir: Path, scratch: Path, backend: str):
    bin_path = src_dir / "pytorch_model.bin"
    if not bin_path.is_file():
        raise SystemExit(f"缺少 {bin_path}")
    sd = load_weights(bin_path, backend, scratch)
    drop = ("cls.", "bert.pooler.", "lm_head.", "bert.embeddings.position_ids")
    sd = {k: v for k, v in sd.items() if not k.startswith(drop)}
    print(f"[bert]   {len(sd)} 张量")
    report_modules(sd, "bert")
    shapes = save_safetensors(out_dir / "bert.safetensors", to_f32(sd))
    copy_if_exists(src_dir / "config.json", out_dir / "bert_config.json")
    for name in ("vocab.txt", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        copy_if_exists(src_dir / name, out_dir / "bert_tokenizer" / name)
    return shapes


def convert_sv(path: Path, out_dir: Path, scratch: Path, backend: str):
    sd = load_weights(path, backend, scratch)
    if "weight" in sd and isinstance(sd["weight"], dict):
        sd = sd["weight"]
    print(f"[sv]     {len(sd)} 张量")
    report_modules(sd, "eres2net")
    return save_safetensors(out_dir / "sv_eres2net.safetensors", to_f32(sd))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def default_paths(gsv_root: Path, model_version: str) -> dict[str, Path]:
    pm = gsv_root / "GPT_SoVITS" / "pretrained_models"
    paths = {
        "gpt": pm / "s1v3.ckpt",
        "hubert": pm / "chinese-hubert-base",
        "bert": pm / "chinese-roberta-wwm-ext-large",
    }
    if model_version in ("v2Pro", "v2ProPlus"):
        # 社区包里经常只放其中一个，做一次回退
        candidates = [pm / "v2Pro" / f"s2G{model_version}.pth"]
        candidates += [pm / "v2Pro" / "s2Gv2ProPlus.pth", pm / "v2Pro" / "s2Gv2Pro.pth"]
        paths["sovits"] = next((p for p in candidates if p.is_file()), candidates[0])
        paths["sv"] = pm / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt"
    elif model_version == "v2":
        paths["sovits"] = pm / "gsv-v2final-pretrained" / "s2G2333k.pth"
    elif model_version == "v4":
        paths["sovits"] = pm / "gsv-v4-pretrained" / "s2Gv4.pth"
        paths["vocoder"] = pm / "gsv-v4-pretrained" / "vocoder.pth"
    elif model_version == "v3":
        paths["sovits"] = pm / "s2Gv3.pth"
        paths["vocoder"] = pm / "models--nvidia--bigvgan_v2_24khz_100band_256x"
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description="GPT-SoVITS -> safetensors")
    ap.add_argument("--gsv-root", type=Path, help="GPT-SoVITS 仓库根目录")
    ap.add_argument("--out", type=Path, required=True, help="输出目录")
    ap.add_argument("--version", default="v2Pro", help="v1/v2/v2Pro/v2ProPlus/v3/v4")
    ap.add_argument("--model-version", default=None, help="覆盖自动识别的 model_version")
    ap.add_argument("--gpt", type=Path)
    ap.add_argument("--sovits", type=Path)
    ap.add_argument("--vocoder", type=Path)
    ap.add_argument("--hubert", type=Path)
    ap.add_argument("--bert", type=Path)
    ap.add_argument("--sv", type=Path)
    ap.add_argument("--backend", choices=["native", "torch"], default="native")
    ap.add_argument("--keep-enc-q", action="store_true", help="保留训练用的 enc_q 分支")
    ap.add_argument("--scratch", type=Path, default=None, help="修正文件头时的临时目录")
    args = ap.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch: Path = args.scratch or (out_dir / ".scratch")

    found: dict[str, Path] = {
        "gpt": args.gpt,
        "sovits": args.sovits,
        "vocoder": args.vocoder,
        "hubert": args.hubert,
        "bert": args.bert,
        "sv": args.sv,
    }
    if args.gsv_root is not None:
        for key, value in default_paths(args.gsv_root, args.version).items():
            if found.get(key) is None:
                found[key] = value
    found = {k: v for k, v in found.items() if v is not None}

    if "sovits" not in found:
        raise SystemExit("必须提供 --sovits（或用 --gsv-root 自动定位）")

    manifest: dict = {
        "format": "aha-gpt-sovits",
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend": args.backend,
        "source": {},
        "tensors": {},
    }

    # ---- s2G 合成器 ------------------------------------------------------
    sovits_path = found["sovits"]
    arch_version, model_version, is_lora = detect_sovits_version(sovits_path)
    if args.model_version:
        model_version = args.model_version
    print(f"[sovits] {sovits_path.name}: arch={arch_version} model={model_version} lora={is_lora}")
    if is_lora:
        raise SystemExit("检测到 LoRA 权重：请先并入底模（GPT-SoVITS 的 merge_and_unload）再转换。")

    dump = load_weights(sovits_path, args.backend, scratch)
    for key in NON_WEIGHT_KEYS:
        dump.pop(key, None)
    sd = dump.get("weight", dump)
    sd, merged = merge_weight_norm(sd)

    dropped: list[str] = []
    kept: dict[str, np.ndarray] = {}
    for key, arr in sd.items():
        if key.startswith("enc_q.") and not args.keep_enc_q:
            dropped.append(key)
            continue
        kept[key] = arr
    unknown = [k for k in kept if not k.startswith(KEEP_PREFIXES)]
    if unknown:
        print(f"  ! 未归类的张量 {len(unknown)} 个，例: {unknown[:4]}")
    kept = {k: v for k, v in kept.items() if k.startswith(KEEP_PREFIXES)}

    print(f"[sovits] weight_norm 合并 {merged} 组（归一化维度分布: {DIM_STATS}），丢弃 enc_q {len(dropped)} 个张量")
    leftover = [k for k in kept if k.endswith((".weight_g", ".weight_v"))]
    if leftover:
        raise SystemExit(f"仍有未合并的 weight_norm：{leftover[:5]}")
    manifest["tensors"]["sovits"] = save_safetensors(
        out_dir / "sovits.safetensors", to_f32(kept)
    )
    manifest["module_params"] = {"sovits": report_modules(kept, "sovits")}

    hps = dump.get("config")
    if isinstance(hps, dict):
        hps = json.loads(json.dumps(hps, default=str))
        hps.setdefault("model", {})["version"] = model_version
        # 强制 25Hz 语义帧率：TTS.py 里也是这么覆盖的
        hps["model"]["semantic_frame_rate"] = "25hz"
        manifest["hps"] = hps
        manifest["audio"] = {
            "sampling_rate": hps["data"]["sampling_rate"],
            "hop_length": hps["data"]["hop_length"],
            "filter_length": hps["data"]["filter_length"],
            "semantic_frame_rate": "25hz",
        }
    manifest["model_version"] = model_version
    manifest["arch_version"] = arch_version
    manifest["is_v2pro"] = model_version in V2PRO_VERSIONS
    manifest["needs_vocoder"] = model_version in V3V4_VERSIONS
    manifest["source"]["sovits"] = str(sovits_path)

    # ---- s1 GPT (AR) -----------------------------------------------------
    if "gpt" in found:
        gpt_path = found["gpt"]
        gpt = load_weights(gpt_path, args.backend, scratch)
        gpt_sd = gpt.get("weight", gpt)
        gpt_sd = {
            (k[len("model.") :] if k.startswith("model.") else k): v for k, v in gpt_sd.items()
        }
        manifest["tensors"]["gpt"] = save_safetensors(
            out_dir / "gpt.safetensors", to_f32(gpt_sd)
        )
        manifest["module_params"]["gpt"] = report_modules(gpt_sd, "gpt")
        if "config" in gpt:
            manifest["t2s_config"] = json.loads(json.dumps(gpt["config"], default=str))
        manifest["source"]["gpt"] = str(gpt_path)
        print(f"[gpt]    {gpt_path.name}: {len(gpt_sd)} 张量")

    # ---- 声码器（v3 / v4） ----------------------------------------------
    if "vocoder" in found:
        voc_path = found["vocoder"]
        if voc_path.is_dir():  # v3 是 HF 目录（BigVGAN）
            sd = load_weights(voc_path / "pytorch_model.bin", args.backend, scratch)
            sd = {k: v for k, v in sd.items() if k.startswith(("conv_pre", "ups", "resblocks", "conv_post"))}
        else:  # v4 是裸 state_dict
            sd = load_weights(voc_path, args.backend, scratch)
            sd = sd.get("weight", sd)
            sd, merged = merge_weight_norm(sd)
            print(f"[vocoder] {voc_path.name}: 合并 {merged} 组 weight_norm")
        manifest["tensors"]["vocoder"] = save_safetensors(
            out_dir / "vocoder.safetensors", to_f32(sd)
        )
        manifest["source"]["vocoder"] = str(voc_path)

    # ---- 特征模型 --------------------------------------------------------
    if "hubert" in found:
        manifest["tensors"]["hubert"] = convert_hubert(found["hubert"], out_dir, scratch, args.backend)
        manifest["source"]["hubert"] = str(found["hubert"])
    if "bert" in found:
        manifest["tensors"]["bert"] = convert_bert(found["bert"], out_dir, scratch, args.backend)
        manifest["source"]["bert"] = str(found["bert"])

    # ---- v2Pro 的说话人向量 ---------------------------------------------
    if manifest["is_v2pro"]:
        if "sv" not in found:
            raise SystemExit("v2Pro / v2ProPlus 必须提供 --sv（eres2netv2 权重）")
        manifest["tensors"]["sv"] = convert_sv(found["sv"], out_dir, scratch, args.backend)
        manifest["source"]["sv"] = str(found["sv"])

    # ---- 符号表 ----------------------------------------------------------
    if args.gsv_root is not None:
        symbols = load_symbols(args.gsv_root, arch_version)
        if symbols:
            (out_dir / "symbols.json").write_text(
                json.dumps(symbols, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            manifest["num_symbols"] = len(symbols)
            vocab = manifest["tensors"]["sovits"].get("enc_p.text_embedding.weight")
            if vocab and vocab[0] != len(symbols):
                print(f"  ! 注意：text_embedding {vocab[0]} != symbols {len(symbols)}")

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    shutil.rmtree(scratch, ignore_errors=True)

    total = sum(p.stat().st_size for p in out_dir.glob("*.safetensors"))
    print(f"\n完成 -> {out_dir}  ({total / 1024 / 1024:.1f} MiB)")
    for name in ("gpt", "sovits", "vocoder", "hubert", "bert", "sv"):
        if name in manifest["tensors"]:
            print(f"  {name:<8} {len(manifest['tensors'][name])} tensors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
