#!/usr/bin/env python3
"""导出 GPT-SoVITS P2 的 torch 参考特征，供 Rust 侧逐层对齐。

**需要在能跑 GPT-SoVITS 的环境执行**（torch + transformers + numpy）。

产出 `ref_feats.npz`：

    bert_ids          (S,)        BERT 输入 token ids（含 [CLS]/[SEP]）
    bert_layer21      (1, S, 1024) 第 21 层输出（= hidden_states[-3]）
    hubert_feats      (1, T', 768) HuBERT last_hidden_state
    wav               (N,)        输入波形（16 kHz f32），保证两边输入完全一致
    text                          对应的文本

用法::

    python export_reference_features.py --gsv-root D:/GPT-SoVITS --out ref_feats.npz

之后在 Rust 侧：

    AHA_GPT_SOVITS_DIR=... AHA_GSV_REF_FEATURES=ref_feats.npz \
      cargo test --features gpt-sovits --test test_gpt_sovits_features -- --nocapture
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


TEXT = "今天天气不错，我们出去走走吧。"


def make_wav(sample_rate: int = 16_000, seconds: float = 1.0, freq: float = 440.0) -> np.ndarray:
    """和 Rust 测试里完全一致的确定性波形。"""
    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    return (np.sin(2 * np.pi * freq * t) * 0.3).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gsv-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("ref_feats.npz"))
    ap.add_argument("--text", default=TEXT)
    args = ap.parse_args()

    gsv = args.gsv_root / "GPT_SoVITS"
    sys.path.insert(0, str(gsv))
    pm = gsv / "pretrained_models"

    from transformers import (
        AutoModelForMaskedLM,
        AutoTokenizer,
        HubertModel,
        Wav2Vec2FeatureExtractor,
    )

    # ---- BERT 第 21 层 ---------------------------------------------------
    bert_dir = str(pm / "chinese-roberta-wwm-ext-large")
    tokenizer = AutoTokenizer.from_pretrained(bert_dir, local_files_only=True)
    # 强制 fp32：配置里写的是 float16，但 fp16 前向会带来 ~3% 的数值偏差，不能当基准
    bert = (
        AutoModelForMaskedLM.from_pretrained(
            bert_dir, local_files_only=True, torch_dtype=torch.float32
        )
        .float()
        .eval()
    )
    inputs = tokenizer(args.text, return_tensors="pt")
    with torch.no_grad():
        out = bert(**inputs, output_hidden_states=True)
    layer21 = out["hidden_states"][-3][0]  # (S, 1024)
    bert_ids = inputs["input_ids"][0].numpy()
    print(f"bert: ids={bert_ids.shape} layer21={tuple(layer21.shape)}")

    # ---- HuBERT ----------------------------------------------------------
    hubert_dir = str(pm / "chinese-hubert-base")
    fe = Wav2Vec2FeatureExtractor.from_pretrained(hubert_dir, local_files_only=True)
    hubert = (
        HubertModel.from_pretrained(
            hubert_dir, local_files_only=True, torch_dtype=torch.float32
        )
        .float()
        .eval()
    )
    print(f"  bert dtype={next(bert.parameters()).dtype}  hubert dtype={next(hubert.parameters()).dtype}")
    wav = make_wav()
    # 注意：GPT-SoVITS 的 _set_prompt_semantic 是 `cnhuhbert_model.model(wav16k)`，
    # 绕过了 feature_extractor（不做 mean/var 归一化）。这里必须一致，否则基准是错的。
    raw = torch.from_numpy(wav)[None, :]
    normalized = fe(wav, return_tensors="pt", sampling_rate=16_000).input_values
    with torch.no_grad():
        hf = hubert(raw)["last_hidden_state"][0]  # (T', 768)
        hf_norm = hubert(normalized)["last_hidden_state"][0]
    print(
        f"  原始波形 cos(归一化前,归一化后) = "
        f"{torch.nn.functional.cosine_similarity(hf.flatten(), hf_norm.flatten(), dim=0).item():.6f}"
        f"  std raw={hf.std().item():.4f} norm={hf_norm.std().item():.4f}"
    )
    print(f"hubert: {tuple(hf.shape)}")

    np.savez(
        args.out,
        bert_ids=bert_ids,
        bert_layer21=layer21.numpy(),
        hubert_feats=hf.numpy(),
        wav=wav,
        text=np.array(args.text),
    )
    print(f"-> {args.out}")

    # ---- T2S (AR) 第一步 logits + 前 10 步 greedy -------------------------
    try:
        dump_ar(args.gsv_root, layer21[None, :, :].transpose(1, 2), args.out)
    except Exception as exc:  # AR 需要 torchmetrics 等额外依赖，失败不影响 BERT/HuBERT
        print(f"跳过 AR 导出：{exc}")
    return 0


def dump_ar(gsv_root: Path, bert_feature, out_path: Path, steps: int = 10) -> None:
    """复用 s1 权重，导出第一步 logits 和前几步 greedy token，用于 Rust 对齐。

    输入是固定的（文本 id 1..N、提示 code 100..119），两边必须完全一致。
    """
    gsv = gsv_root / "GPT_SoVITS"
    sys.path.insert(0, str(gsv))

    ckpt = gsv / "pretrained_models" / "s1v3.ckpt"
    if not ckpt.is_file():
        print(f"跳过 AR 导出：找不到 {ckpt}")
        return

    from AR.models.t2s_model import Text2SemanticDecoder

    dump = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = dump["config"]
    model = Text2SemanticDecoder(cfg).eval()
    state = {k[len("model.") :]: v for k, v in dump["weight"].items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  ! AR 权重缺失 {len(missing)} 个：{missing[:3]}")

    text_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    prompt = torch.tensor([[100 + i for i in range(20)]], dtype=torch.long)
    bert_feature = bert_feature[:, : text_ids.shape[1], :]

    with torch.no_grad():
        x = model.ar_text_embedding(text_ids)
        x = x + model.bert_proj(bert_feature)
        x = model.ar_text_position(x)
        y_emb = model.ar_audio_embedding(prompt)
        y_pos = model.ar_audio_position(y_emb)
        xy = torch.cat([x, y_pos], dim=1)

        x_len, y_len = text_ids.shape[1], prompt.shape[1]
        x_mask = torch.zeros((x_len, x_len), dtype=torch.bool)
        x_mask_pad = torch.nn.functional.pad(x_mask, (0, y_len), value=True)
        y_mask = torch.nn.functional.pad(
            torch.triu(torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1), (x_len, 0), value=False
        )
        attn_mask = torch.cat([x_mask_pad, y_mask], dim=0).unsqueeze(0).unsqueeze(0)

        xy_dec, k_cache, v_cache = model.t2s_transformer.process_prompt(xy, attn_mask, None)
        logits = model.ar_predict_layer(xy_dec[:, -1])  # (1, 1025)
        first_logits = logits.clone()

        greedy = []
        y = prompt
        for idx in range(steps):
            cur = logits[:, :-1] if idx < 11 else logits
            token = torch.argmax(cur, dim=-1, keepdim=True)
            greedy.append(int(token.item()))
            y = torch.cat([y, token], dim=1)
            next_emb = model.ar_audio_embedding(y[:, -1:])
            xy_pos = next_emb + model.ar_audio_position.alpha * model.ar_audio_position.pe[:, y_len + idx]
            xy_dec, k_cache, v_cache = model.t2s_transformer.decode_next_token(xy_pos, k_cache, v_cache)
            logits = model.ar_predict_layer(xy_dec[:, -1])

    np.savez(
        out_path.with_name("ref_ar.npz"),
        text_ids=text_ids[0].numpy(),
        prompt=prompt[0].numpy(),
        # (1024, S)：与 Rust 侧 `Text2Semantic::infer` 的入参一致
        bert_feature=bert_feature[0].transpose(0, 1).numpy(),
        first_logits=first_logits[0].numpy(),
        greedy=np.array(greedy),
    )
    print(f"-> {out_path.with_name('ref_ar.npz')}  greedy={greedy[:5]}...")


if __name__ == "__main__":
    sys.exit(main())
