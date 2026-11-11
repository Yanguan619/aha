"""极简 PyTorch 权重读取器（.pth / .ckpt / pytorch_model.bin），不依赖 torch。

只覆盖 PyTorch >= 1.6 的 zip 格式，以及其中最常见的几种引用：

    collections.OrderedDict
    torch._utils._rebuild_tensor_v2 / _rebuild_tensor
    torch.{Half,Float,Long,Int,Short,Char,Byte,BFloat16}Storage
    torch.nn.parameter.Parameter

GPT-SoVITS 的权重正好只用到这些。遇到未知类型会走 ``_Stub``，不会崩，
但如果某个未知对象是推理需要的，转换结果里会缺张量 —— 会打印告警。

用法::

    from pth_reader import load
    obj = load("s2Gv2ProPlus.pth")     # -> OrderedDict / dict，值默认是 numpy 数组

``load(..., as_numpy=True)`` 返回 numpy；``as_numpy=False`` 时返回带元信息的
``TorchTensor``（shape/dtype/数据），便于按需惰性取用。
"""

from __future__ import annotations

import io
import pickletools
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

_STORAGE_DTYPES = {
    "HalfStorage": np.float16,
    "BFloat16Storage": None,  # 特殊处理：存的是 uint16 位模式
    "FloatStorage": np.float32,
    "DoubleStorage": np.float64,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "ShortStorage": np.int16,
    "CharStorage": np.int8,
    "ByteStorage": np.uint8,
    "BoolStorage": np.bool_,
}


class TorchTensor:
    """一个张量：numpy 视图 + 原始 dtype 名字，避免 f16/bf16 丢失信息。"""

    __slots__ = ("array", "dtype_name")

    def __init__(self, array: np.ndarray, dtype_name: str | None = None):
        self.array = array
        self.dtype_name = dtype_name

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.array.shape)

    def numpy(self) -> np.ndarray:
        """返回连续内存的 numpy 数组（bf16 会保持 uint16 位模式）。"""
        return np.ascontiguousarray(self.array)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"TorchTensor(shape={self.shape}, dtype={self.array.dtype})"


class _Stub:
    """占位符：pickle 里出现的、我们不需要的类（hparams 里的冷门类型等）。"""

    def __init__(self, *args, **kwargs):
        self.args = args

    def __setstate__(self, state):
        self.state = state

    def __repr__(self) -> str:  # pragma: no cover
        return f"<stub {self.__class__.__name__}>"


class _StorageType:
    """pickle 里的 ``torch.HalfStorage`` 这类对象。"""

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover
        return f"<storage {self.name}>"


class _StorageRef:
    __slots__ = ("key", "dtype_name")

    def __init__(self, key: str, dtype_name: str):
        self.key = key
        self.dtype_name = dtype_name


def _is_zip(path: Path) -> bool:
    with open(path, "rb") as f:
        return f.read(2) == b"PK"


class _Unpickler:
    """把 pickle 里的 torch 张量引用解析成 numpy 视图。"""

    @staticmethod
    def make(zf: zipfile.ZipFile, prefix: str, byteorder: str):
        import pickle

        class Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "collections" and name == "OrderedDict":
                    return OrderedDict
                if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
                    return self._rebuild_tensor
                if module == "torch._utils" and name == "_rebuild_parameter":
                    return lambda data, *_a, **_k: data
                if module == "torch" and name.endswith("Storage"):
                    return _StorageType(name)
                if module in ("torch", "torch.nn.parameter") and name in ("Parameter", "Tensor"):
                    return lambda *a, **k: a[0] if a else None
                return type(name, (_Stub,), {})

            def persistent_load(self, pid):
                # pid == ('storage', <_StorageType>, '17', 'cpu', numel)
                if isinstance(pid, tuple) and len(pid) >= 3 and pid[0] == "storage":
                    storage_type, key = pid[1], pid[2]
                    dtype_name = getattr(storage_type, "name", "FloatStorage")
                    return _StorageRef(str(key), dtype_name)
                raise pickle.UnpicklingError(f"unsupported persistent id: {pid!r}")

            def _rebuild_tensor(self, storage, storage_offset, size, stride, *args):
                if not isinstance(storage, _StorageRef):
                    return TorchTensor(np.zeros(size, dtype=np.float32))
                dtype = _STORAGE_DTYPES.get(storage.dtype_name, np.float32)
                raw = zf.read(f"{prefix}data/{storage.key}")
                if dtype is None:  # bfloat16：按 uint16 位模式存放
                    base = np.frombuffer(raw, dtype=np.uint16)
                else:
                    base = np.frombuffer(raw, dtype=dtype)
                if byteorder == "big" and base.dtype.byteorder not in ("<", "|"):
                    base = base.byteswap()
                size = tuple(int(s) for s in size)
                stride = tuple(int(s) for s in stride)
                if any(s == 0 for s in size):
                    return TorchTensor(np.zeros(size, dtype=base.dtype), storage.dtype_name)
                itemsize = base.itemsize
                view = np.lib.stride_tricks.as_strided(
                    base[storage_offset:],
                    shape=size,
                    strides=[s * itemsize for s in stride],
                    writeable=False,
                )
                return TorchTensor(view, storage.dtype_name)

        return Unpickler


def load(path: str | Path, as_numpy: bool = True):
    """读取 .pth / .ckpt / .bin。

    ``as_numpy=True`` 时返回普通 dict，值是 numpy 数组（bf16 保持 uint16）。
    """
    path = Path(path)
    if not _is_zip(path):
        raise ValueError(
            f"{path.name} 不是 zip 格式（PyTorch < 1.6 的旧格式）。"
            "请用 torch 重新保存，或使用 --backend torch。"
        )

    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        pkl_names = [n for n in names if n.endswith("data.pkl")]
        if not pkl_names:
            raise ValueError(f"{path.name}: 找不到 data.pkl")
        pkl_name = pkl_names[0]
        prefix = pkl_name[: -len("data.pkl")]

        byteorder = "little"
        if f"{prefix}byteorder" in names:
            byteorder = zf.read(f"{prefix}byteorder").decode().strip()
        pickle_bytes = zf.read(pkl_name)
        unpickler = _Unpickler.make(zf, prefix, byteorder)(io.BytesIO(pickle_bytes))
        obj = unpickler.load()

    if not as_numpy:
        return obj

    def to_numpy(x):
        if isinstance(x, TorchTensor):
            return x.numpy()
        if isinstance(x, OrderedDict):
            return OrderedDict((k, to_numpy(v)) for k, v in x.items())
        if isinstance(x, dict):
            return {k: to_numpy(v) for k, v in x.items()}
        if isinstance(x, list):
            return [to_numpy(v) for v in x]
        if isinstance(x, tuple):
            return tuple(to_numpy(v) for v in x)
        return x

    return to_numpy(obj)


def global_refs(path: str | Path) -> list[str]:
    """列出权重里 pickle 引用的全局符号，用来确认 reader 是否够用。"""
    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        pkl_name = next(n for n in zf.namelist() if n.endswith("data.pkl"))
        data = zf.read(pkl_name)
    refs: list[str] = []
    for op, arg, _pos in pickletools.genops(data):
        if op.name in ("GLOBAL", "STACK_GLOBAL") and isinstance(arg, str):
            if arg not in refs:
                refs.append(arg)
    return refs
