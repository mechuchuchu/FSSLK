"""Several named CPU tensors in a single protocol frame."""

from __future__ import annotations

from typing import Any

import torch

from .protocol import MAX_PAYLOAD_BYTES
from .tensor import decode_tensor, encode_tensor

MAX_BUNDLE_ITEMS = 4096
MAX_NAME_LENGTH = 128


def encode_bundle(tensors: dict[str, torch.Tensor]) -> tuple[list[dict[str, Any]], bytes]:
    """Return per-tensor metadata and one payload with the tensors back to back."""
    if len(tensors) > MAX_BUNDLE_ITEMS:
        raise ValueError(f"a bundle holds at most {MAX_BUNDLE_ITEMS} tensors")
    meta: list[dict[str, Any]] = []
    chunks: list[bytes] = []
    for name, tensor in tensors.items():
        if not isinstance(name, str) or not name or len(name) > MAX_NAME_LENGTH:
            raise ValueError(f"invalid tensor name: {name!r}")
        tensor_meta, payload = encode_tensor(tensor)
        meta.append({"name": name, **tensor_meta, "nbytes": len(payload)})
        chunks.append(payload)
    payload = b"".join(chunks)
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError(f"bundle exceeds the {MAX_PAYLOAD_BYTES}-byte frame limit")
    return meta, payload


def decode_bundle(meta: Any, payload: bytes) -> dict[str, torch.Tensor]:
    """Rebuild a bundle from untrusted metadata and payload."""
    if not isinstance(meta, list) or len(meta) > MAX_BUNDLE_ITEMS:
        raise ValueError("bundle metadata must be a list of tensors")
    tensors: dict[str, torch.Tensor] = {}
    offset = 0
    for entry in meta:
        if not isinstance(entry, dict):
            raise ValueError("bundle entry must be an object")
        name, nbytes = entry.get("name"), entry.get("nbytes")
        if not isinstance(name, str) or not name or len(name) > MAX_NAME_LENGTH or name in tensors:
            raise ValueError(f"invalid or duplicate tensor name: {name!r}")
        if not isinstance(nbytes, int) or isinstance(nbytes, bool) or nbytes < 0:
            raise ValueError(f"tensor {name} has an invalid size")
        if offset + nbytes > len(payload):
            raise ValueError("bundle payload is shorter than its metadata")
        tensors[name] = decode_tensor(entry, payload[offset : offset + nbytes])
        offset += nbytes
    if offset != len(payload):
        raise ValueError("bundle payload is longer than its metadata")
    return tensors


def split_bundles(
    tensors: dict[str, torch.Tensor],
    max_bytes: int = 32 * 1024 * 1024,
    max_items: int = 256,
) -> list[dict[str, torch.Tensor]]:
    """Group tensors into bundles that each fit one frame.

    A tensor larger than ``max_bytes`` travels alone, as long as it still fits a
    frame; larger ones cannot be sent by this protocol yet.
    """
    bundles: list[dict[str, torch.Tensor]] = []
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0
    for name, tensor in tensors.items():
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes > MAX_PAYLOAD_BYTES:
            raise ValueError(
                f"tensor {name} is {nbytes} bytes, more than one {MAX_PAYLOAD_BYTES}-byte frame"
            )
        if current and (current_bytes + nbytes > max_bytes or len(current) >= max_items):
            bundles.append(current)
            current, current_bytes = {}, 0
        current[name] = tensor
        current_bytes += nbytes
    if current:
        bundles.append(current)
    return bundles
