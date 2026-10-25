"""Small safe JSON/base64 codec for CPU float32 tensors used by demo workers."""

from __future__ import annotations

import base64
import binascii
import math
import sys
from typing import Any

import torch


def encode_float32(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    raw = bytes(value.view(torch.uint8).reshape(-1).tolist())
    return {
        "dtype": "float32",
        "shape": list(value.shape),
        "data": base64.b64encode(raw).decode("ascii"),
    }


def decode_float32(payload: Any) -> torch.Tensor:
    if not isinstance(payload, dict) or payload.get("dtype") != "float32":
        raise ValueError("tensor payload must declare float32 dtype")
    shape = payload.get("shape")
    if (
        not isinstance(shape, list)
        or any(not isinstance(size, int) or size < 0 for size in shape)
    ):
        raise ValueError("tensor shape must be a list of non-negative integers")
    data = payload.get("data")
    if not isinstance(data, str):
        raise ValueError("tensor data must be base64 text")
    try:
        raw = base64.b64decode(data, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("tensor data is not valid base64") from exc
    expected_bytes = math.prod(shape) * torch.tensor([], dtype=torch.float32).element_size()
    if len(raw) != expected_bytes:
        raise ValueError("tensor data length does not match its shape")
    if sys.byteorder != "little":
        raise RuntimeError("the demo tensor codec currently requires a little-endian host")
    if expected_bytes == 0:
        return torch.empty(shape, dtype=torch.float32)
    storage = bytearray(raw)
    return torch.frombuffer(storage, dtype=torch.float32).clone().reshape(shape)
