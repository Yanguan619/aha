#!/usr/bin/env python3
"""把 GPT-SoVITS v2Pro 的 SoVITS 解码链（ref_enc + enc_p + flow + HiFiGAN）导出成定长 ONNX。

只导出"非自回归"的部分：GPT(T2S) 仍然逐 token 自回归，留在 CPU；这里导出的图
吃 (codes, text, refer, sv_emb) 直接产波形，正好可以交给 QNN(他 HTP) 跑。

用法（宿主，普通 python 即可，不需要 QNN SDK）:
    python scripts/gpt_sovits_export_onnx.py \
        --ckpt E:/weights/gptsovits-ckpt/s2Gv2Pro.pth \
        --src  E:/weights/gptsovits-src/GPT-SoVITS-main \
        --out  E:/weights/gptsovits-onnx/vits.onnx \
        --codes 64 --text 96 --refer 704

导出后用 onnxruntime CPU 跑一遍自检，再交给 scripts 里的量化/转换工具。
"""

from __future__ import annotations

import argparse
import os
import sys
import types


def _stub_f5_tts() -> None:
    """models_onnx 顶部 `from f5_tts.model import DiT` 只在 v3/v4 用到。

    v2Pro 用不到 DiT，但 import 是模块级的，所以这里塞一个占位模块，避免装 f5-tts。
    """
    if "f5_tts" in sys.modules:
        return
    import torch.nn as nn

    pkg = types.ModuleType("f5_tts")
    model = types.ModuleType("f5_tts.model")

    class DiT(nn.Module):  # noqa: D101 - 占位
        def __init__(self, *a, **kw):  # noqa: D107
            super().__init__()

    model.DiT = DiT
    pkg.model = model  # type: ignore[attr-defined]
    sys.modules["f5_tts"] = pkg
    sys.modules["f5_tts.model"] = model


class _VitsExport:
    """把 SynthesizerTrn 包成"定长、无随机"的一层，便于导出成静态图。"""

    def __init__(self, module_file, codes: int, text: int, refer_frames: int, noise_scale: float = 0.0):
        self.codes = codes
        self.text = text
        self.refer_frames = refer_frames
        self.noise_scale = noise_scale
        self.module_file = module_file


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="导出 GPT-SoVITS v2Pro 的 SoVITS 解码链为定长 ONNX")
    p.add_argument("--ckpt", required=True, help="s2Gv2Pro.pth 路径")
    p.add_argument("--src", required=True, help="GPT-SoVITS 源码根目录（含 GPT_SoVITS/）")
    p.add_argument("--out", required=True, help="输出 onnx 路径")
    p.add_argument("--codes", type=int, default=64, help="语义 token 固定帧数（25Hz，64 帧≈2.56s）")
    p.add_argument("--text", type=int, default=96, help="音素固定长度（不足补 0）")
    p.add_argument("--refer", type=int, default=704, help="参考音频频谱固定帧数")
    p.add_argument("--noise-scale", type=float, default=0.0, help="0 = 确定性（不采样噪声）")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--verify", action="store_true", help="导出后用 onnxruntime 自检")
    return p


def main() -> int:
    args = build_parser().parse_args()

    _stub_f5_tts()
    src = os.path.abspath(args.src)
    sys.path.insert(0, os.path.join(src, "GPT_SoVITS"))

    import torch
    import torch.nn as nn

    from module.models_onnx import SynthesizerTrn

    def dict_to_attr(d):
        if isinstance(d, dict):
            return types.SimpleNamespace(**{k: dict_to_attr(v) for k, v in d.items()})
        return d

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    hps = dict_to_attr(sd["config"])
    hps.model.version = "v2Pro"
    hps.model.semantic_frame_rate = "25hz"

    model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **vars(hps.model),
    )
    missing, unexpected = model.load_state_dict(sd["weight"], strict=False)
    model.eval()

    gin = hps.model.gin_channels
    sv_dim = model.sv_emb.in_features

    class Wrap(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, codes, text, refer, sv_emb):
            # 复刻 SynthesizerTrn.forward 的 v2Pro 分支，但把 `ones_like(refer[:, :1, :])`
            # 换成常量 mask —— 后者会留下 Slice+动态 shape，QNN 转换器解不出来
            # （报 `Unable to broadcast shapes: [0,0,0] and [1,1,704]`）。
            m = self.m
            mask = torch.ones(1, 1, 1, dtype=refer.dtype)
            ge = m.ref_enc(refer * mask, mask)
            if m.is_v2pro:
                sv = m.sv_emb(sv_emb)
                ge = m.prelu(ge + sv.unsqueeze(-1))
            quantized = m.quantizer.decode(codes)
            if m.semantic_frame_rate == "25hz":
                dq = torch.cat([quantized, quantized]).permute(1, 2, 0)
                quantized = dq.contiguous().view(1, m.ssl_dim, -1)
            ge_in = m.ge_to512(ge.transpose(2, 1)).transpose(2, 1) if m.is_v2pro else ge
            x, m_p, logs_p, y_mask = m.enc_p(quantized, text, ge_in, 1.0)
            # noise_scale=0 时不要画随机数：否则图上会留下 RandomNormalLike，
            # QNN 转换器没有这个算子的 translation（报 `No translation registered for op type onnx_randomnormallike.`）。
            z_p = m_p
            if args.noise_scale:
                z_p = z_p + torch.randn_like(m_p) * torch.exp(logs_p) * args.noise_scale
            z = m.flow(z_p, y_mask, g=ge, reverse=True)
            o = m.dec(z * y_mask, g=ge)
            return o[0]

    wrap = Wrap(model).eval()

    # ResidualVectorQuantizer.decode 要的是 [B, n_q, T]（n_q=1），不是 [B, T]
    codes = torch.randint(0, 1024, (1, 1, args.codes), dtype=torch.long)
    text = torch.randint(0, 100, (1, args.text), dtype=torch.long)
    refer = torch.randn(1, args.refer, 704) if False else torch.randn(1, 704, args.refer)
    sv = torch.randn(1, sv_dim)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with torch.no_grad():
        ref_out = wrap(codes, text, refer, sv)
        print(f"torch out shape: {tuple(ref_out.shape)}")
        torch.onnx.export(
            wrap,
            (codes, text, refer, sv),
            args.out,
            input_names=["codes", "text", "refer", "sv_emb"],
            output_names=["audio"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"saved {args.out} = {os.path.getsize(args.out)} bytes")
    print(f"missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    if missing:
        print("  missing sample:", list(missing)[:5])
    if unexpected:
        print("  unexpected sample:", list(unexpected)[:5])

    if args.verify:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        feeds = {
            "codes": codes.numpy(),
            "text": text.numpy(),
            "refer": refer.numpy(),
            "sv_emb": sv.numpy(),
        }
        out = sess.run(None, feeds)[0]
        a = ref_out.numpy().reshape(-1)
        b = out.reshape(-1)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
        cos = float(np.dot(a, b) / denom)
        print(f"onnxruntime out shape: {out.shape}, cos={cos:.6f}, maxdiff={float(np.abs(a - b).max()):.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
