#!/usr/bin/env python3
"""把 SenseVoice 的 fp32 ONNX 转成 Qualcomm QNN (HTP) 能跑的 QDQ 模型。

HTP 的两条硬限制决定了整个流程：

1. **只支持静态 shape** —— 所以先把 ``x`` 固定成 ``[1, num_frames, 560]``；
2. **只支持量化模型**，官方推荐 QDQ 格式、激活 uint8/uint16。

实测（SenseVoice small、SM8850 / HTP v81、320 帧窗口）：

* 激活用 QUInt8 会把注意力打崩 —— 第 0 层 self-attn 之后 cos≈0.18，
  解码结果全是 blank；
* 换成 **QUInt16 激活 + QUInt8 权重** 后 cos≈0.99，转写文本与 fp32 一致。

所以默认值是 ``activation=QUInt16, weight=QUInt8, per_channel=False``。

注意 SenseVoice 的 logits 长度是「帧数 + 4」，预处理/量化后会按帧数写成静态值，
这里统一改回 ``num_frames + 4``。

用法::

    python scripts/quantize_sense_voice_qnn.py \\
        --model-dir E:/weights/sense-voice \\
        --num-frames 320 \\
        --wav assets/audio/jiangjiang.wav \\
        --wav E:/weights/sense-voice/calib-wavs

产物：``<model-dir>/model.qnn.onnx``（权重走外部存储时另有 ``.data``）。
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.execution_providers.qnn import qnn_preprocess_model
from onnxruntime.quantization.execution_providers.qnn import preprocess as qnn_preprocess

from sense_voice_feats import feats16k, load_audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="目录里要有 model.onnx（fp32）与 model.int8.onnx（读元数据用）",
    )
    parser.add_argument("--num-frames", type=int, default=320, help="固定帧数")
    parser.add_argument(
        "--wav",
        action="append",
        default=[],
        help="校准音频，可给文件或目录（可重复）",
    )
    parser.add_argument("--language", type=int, default=3, help="校准用的 language id")
    parser.add_argument("--text-norm", type=int, default=14, help="校准用的 text_norm id")
    return parser.parse_args()


def load_cmvn(model_dir: str) -> tuple[np.ndarray, np.ndarray]:
    int8 = onnx.load(os.path.join(model_dir, "model.int8.onnx"), load_external_data=False)
    meta = {kv.key: kv.value for kv in int8.metadata_props}
    neg_mean = np.array([float(v) for v in meta["neg_mean"].split(",")], np.float32)
    inv_stddev = np.array([float(v) for v in meta["inv_stddev"].split(",")], np.float32)
    return neg_mean, inv_stddev


def collect_wavs(specs: list[str]) -> list[str]:
    wavs: list[str] = []
    for spec in specs:
        if os.path.isdir(spec):
            wavs.extend(sorted(glob.glob(os.path.join(spec, "*.wav"))))
        elif os.path.exists(spec):
            wavs.append(spec)
        else:
            raise SystemExit(f"校准音频不存在: {spec}")
    return wavs


def build_samples(
    wavs: list[str],
    num_frames: int,
    neg_mean: np.ndarray,
    inv_stddev: np.ndarray,
) -> list[np.ndarray]:
    samples: list[np.ndarray] = []
    for wav in wavs:
        feat = feats16k(load_audio(wav), neg_mean, inv_stddev)
        if len(feat) < num_frames:
            padded = np.zeros((num_frames, feat.shape[1]), np.float32)
            padded[: len(feat)] = feat
            samples.append(padded)
            continue
        step = max(1, num_frames // 8)
        for start in range(0, len(feat) - num_frames + 1, step):
            samples.append(feat[start : start + num_frames])
    if not samples:
        raise SystemExit("没有可用校准数据")
    samples.append(np.zeros((num_frames, samples[0].shape[1]), np.float32))
    print(f"校准样本: {len(samples)} 个, shape=[{num_frames}, {samples[0].shape[1]}]")
    return [sample.astype(np.float32) for sample in samples]


class Reader:
    def __init__(self, samples: list[np.ndarray], language: int, text_norm: int) -> None:
        self.samples = samples
        self.language = language
        self.text_norm = text_norm
        self.index = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.index >= len(self.samples):
            return None
        sample = self.samples[self.index]
        self.index += 1
        return {
            "x": sample[None],
            "x_length": np.array([sample.shape[0]], np.int32),
            "language": np.array([self.language], np.int32),
            "text_norm": np.array([self.text_norm], np.int32),
        }


def _patch_metadata_bug() -> None:
    """绕开 onnxruntime 1.27 的一个 bug。

    ``qnn/preprocess.py`` 里直接 ``dict.update(model.metadata_props)``，
    而 metadata_props 是 StringStringEntryProto 列表，会抛
    ``TypeError: cannot convert dictionary update sequence element #0 to a sequence``。
    """

    def save_and_reload_optimize_model(
        model: onnx.ModelProto, shape_infer: bool = True
    ) -> onnx.ModelProto:
        # 原实现走 quant_pre_process(...)，但那需要落盘路径；这里只做形状推导，
        # 图优化（identity 消除等）对量化不是必需的。
        if shape_infer:
            del model.graph.value_info[:]
            model = onnx.shape_inference.infer_shapes(model)
        props = {kv.key: kv.value for kv in model.metadata_props}
        model.metadata_props.clear()
        for key, value in props.items():
            entry = model.metadata_props.add()
            entry.key = key
            entry.value = value
        return model

    qnn_preprocess.save_and_reload_optimize_model = save_and_reload_optimize_model


def _patch_softmax_range_bug() -> None:
    """绕开 onnxruntime 1.27 的另一个 bug。

    ``base_quantizer.adjust_tensor_ranges`` 会给 Softmax 输出强行设 0..1 的范围，
    但它用节点名去找张量；图被优化过（Softmax 被融合/改名）时该名字已不存在，
    于是抛 ``RuntimeError: Only an existing tensor can be modified, ... is not.``。
    """

    from onnxruntime.quantization import calibrate

    original = calibrate.TensorsData.__setitem__

    def setitem(collection, key, value):
        if key not in collection.data:
            return
        original(collection, key, value)

    calibrate.TensorsData.__setitem__ = setitem


def preprocess(model_dir: str, num_frames: int) -> str:
    _patch_metadata_bug()
    src = os.path.join(model_dir, "model.onnx")
    dst = os.path.join(model_dir, "model.preproc.onnx")
    shapes = [
        ("x", f"1,{num_frames},560"),
        ("x_length", "1"),
        ("language", "1"),
        ("text_norm", "1"),
    ]
    modified = qnn_preprocess_model(src, dst, dynamic_input_shapes=shapes)
    print(f"预处理: modified={modified} -> {dst}")
    return dst


def fix_output_shape(path: str, num_frames: int, vocab: int = 25055) -> None:
    model = onnx.load(path)
    wanted = [1, num_frames + 4, vocab]
    shape = model.graph.output[0].type.tensor_type.shape
    for index, value in enumerate(wanted):
        dim = shape.dim[index]
        dim.Clear()
        dim.dim_value = value
    onnx.save_model(model, path)
    print(f"logits 固定为 {wanted}")


def main() -> None:
    args = parse_args()
    model_dir = os.path.abspath(args.model_dir)
    neg_mean, inv_stddev = load_cmvn(model_dir)
    samples = build_samples(
        collect_wavs(args.wav), args.num_frames, neg_mean, inv_stddev
    )
    src = preprocess(model_dir, args.num_frames)

    dst = os.path.join(model_dir, "model.qnn.onnx")
    _patch_softmax_range_bug()
    quantize_static(
        src,
        dst,
        Reader(samples, args.language, args.text_norm),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt16,
        weight_type=QuantType.QUInt8,
        per_channel=False,
        reduce_range=False,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={"ActivationSymmetric": False},
        use_external_data_format=True,
    )
    fix_output_shape(dst, args.num_frames)
    print(f"完成 -> {dst}")


if __name__ == "__main__":
    main()

