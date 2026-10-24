"""Encoding helpers for CPU tensors sent over the FSSLK wire protocol."""

from __future__ import annotations

import math
import ctypes
from typing import Any

import torch

from .protocol import MAX_PAYLOAD_BYTES


_DTYPE_NAMES = {
    torch.int64: "int64",
    torch.float32: "float32",
}
_DTYPES = {name: dtype for dtype, name in _DTYPE_NAMES.items()}


def encode_tensor(tensor: torch.Tensor) -> tuple[dict[str, Any], bytes]:
    if tensor.device.type != "cpu":
        raise ValueError("only CPU tensors can be sent by this prototype")
    dtype_name = _DTYPE_NAMES.get(tensor.dtype)
    if dtype_name is None:
        raise ValueError(f"unsupported tensor dtype: {tensor.dtype}")

    contiguous = tensor.detach().contiguous().clone(memory_format=torch.contiguous_format)
    expected_size = contiguous.numel() * contiguous.element_size()
    payload = (
        ctypes.string_at(contiguous.untyped_storage().data_ptr(), expected_size)
        if expected_size
        else b""
    )
    if len(payload) != expected_size:
        raise RuntimeError("tensor storage size does not match its shape and dtype")
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError(f"tensor payload exceeds the {MAX_PAYLOAD_BYTES}-byte limit")

    return {"dtype": dtype_name, "shape": list(contiguous.shape)}, payload


def decode_tensor(metadata: dict[str, Any], payload: bytes) -> torch.Tensor:
    dtype_name = metadata.get("dtype")
    dtype = _DTYPES.get(dtype_name) if isinstance(dtype_name, str) else None
    if dtype is None:
        raise ValueError(f"unsupported tensor dtype: {dtype_name!r}")

    shape = metadata.get("shape")
    if not isinstance(shape, list) or len(shape) > 32:
        raise ValueError("tensor shape must be a list with at most 32 dimensions")
    if any(not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in shape):
        raise ValueError("tensor dimensions must be non-negative integers")

    element_count = math.prod(shape)
    expected_size = element_count * torch.empty((), dtype=dtype).element_size()
    if expected_size > MAX_PAYLOAD_BYTES:
        raise ValueError(f"tensor payload exceeds the {MAX_PAYLOAD_BYTES}-byte limit")
    if len(payload) != expected_size:
        raise ValueError(
            f"tensor payload size mismatch: expected {expected_size} bytes, got {len(payload)}"
        )
    if element_count == 0:
        return torch.empty(shape, dtype=dtype, device="cpu")

    tensor = torch.frombuffer(bytearray(payload), dtype=dtype, count=element_count).clone()
    return tensor.reshape(shape)
