"""Allow-listed operations and a data-only encoding of their arguments.

A captured graph is shipped to workers as plain JSON. This module is the whole
trust boundary for *what a worker will run*: an operation is executed only if
its exact name is in ``ALLOWED_OPS``, and its arguments may only be tensor
references, numbers, booleans, strings, ``None``, nested lists, and a few named
torch constants. There is no pickle and no ``eval``; anything else is rejected
while decoding.

Only pure, deterministic, out-of-place operations are listed. Training-mode
dropout (random) is rejected at execution time.
"""

from __future__ import annotations

import functools
import math
from typing import Any, Callable

import torch

MAX_ARG_DEPTH = 8
MAX_ARG_ITEMS = 65_536
MAX_STRING_LENGTH = 64

ALLOWED_OPS = frozenset(
    {
        "getitem",
        # linear algebra
        "aten.linear.default",
        "aten.mm.default",
        "aten.bmm.default",
        "aten.matmul.default",
        "aten.addmm.default",
        "aten.baddbmm.default",
        # attention, normalization, activations
        "aten.scaled_dot_product_attention.default",
        "aten.layer_norm.default",
        "aten.batch_norm.default",
        "aten.relu.default",
        "aten.gelu.default",
        "aten.silu.default",
        "aten.tanh.default",
        "aten.sigmoid.default",
        "aten.softmax.int",
        "aten.log_softmax.int",
        "aten.dropout.default",
        # elementwise
        "aten.add.Tensor",
        "aten.sub.Tensor",
        "aten.mul.Tensor",
        "aten.div.Tensor",
        "aten.neg.default",
        "aten.rsqrt.default",
        "aten.sqrt.default",
        "aten.exp.default",
        "aten.pow.Tensor_Scalar",
        # shape manipulation
        "aten.view.default",
        "aten.reshape.default",
        "aten.permute.default",
        "aten.transpose.int",
        "aten.flatten.using_ints",
        "aten.unsqueeze.default",
        "aten.squeeze.dim",
        "aten.expand.default",
        "aten.clone.default",
        "aten.select.int",
        "aten.slice.Tensor",
        "aten.unbind.int",
        "aten.cat.default",
        "aten.stack.default",
        "aten.split.Tensor",
        "aten.chunk.default",
        # reductions
        "aten.mean.dim",
        "aten.sum.dim_IntList",
        # convolution, pooling, lookup, casts
        "aten.conv1d.default",
        "aten.conv2d.default",
        "aten.max_pool2d.default",
        "aten.avg_pool2d.default",
        "aten.adaptive_avg_pool2d.default",
        "aten.embedding.default",
        "aten.to.dtype",
    }
)

_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64,
    "int32": torch.int32,
    "bool": torch.bool,
}
_MEMORY_FORMATS = {
    "contiguous_format": torch.contiguous_format,
    "preserve_format": torch.preserve_format,
}
_NONFINITE_FLOATS = {"inf": math.inf, "-inf": -math.inf, "nan": math.nan}


class NotEncodable(ValueError):
    """A graph argument cannot be expressed in the data-only encoding."""


# -------------------------------------------------------------------- encoding
def encode_arg(value: Any, depth: int = 0) -> Any:
    """Turn one fx node argument into JSON-safe data."""
    if depth > MAX_ARG_DEPTH:
        raise NotEncodable("argument nesting is too deep")
    if isinstance(value, torch.fx.Node):
        return {"ref": value.name}
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else {"float": repr(float(value))}
    if isinstance(value, str):
        if len(value) > MAX_STRING_LENGTH:
            raise NotEncodable("string argument is too long")
        return value
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_ARG_ITEMS:
            raise NotEncodable("sequence argument is too long")
        return [encode_arg(item, depth + 1) for item in value]
    if isinstance(value, torch.dtype):
        name = str(value).removeprefix("torch.")
        if name not in _DTYPES:
            raise NotEncodable(f"dtype {name} is not allowed")
        return {"dtype": name}
    if isinstance(value, torch.device):
        if value.type != "cpu":
            raise NotEncodable("only cpu tensors are supported")
        return {"device": "cpu"}
    if isinstance(value, torch.memory_format):
        for name, memory_format in _MEMORY_FORMATS.items():
            if value == memory_format:
                return {"memory_format": name}
        raise NotEncodable(f"memory format {value} is not allowed")
    if isinstance(value, torch.layout) and value == torch.strided:
        return {"layout": "strided"}
    raise NotEncodable(f"cannot encode a {type(value).__name__} argument")


def encode_call(node: torch.fx.Node) -> dict[str, Any] | None:
    """Encode a node's arguments, or return ``None`` if some cannot be shipped."""
    try:
        return {
            "args": [encode_arg(arg) for arg in node.args],
            "kwargs": {name: encode_arg(arg) for name, arg in node.kwargs.items()},
        }
    except NotEncodable:
        return None


# -------------------------------------------------------------------- decoding
def decode_arg(value: Any, resolve_ref: Callable[[str], Any], depth: int = 0) -> Any:
    """Rebuild an argument from untrusted data, rejecting anything unexpected."""
    if depth > MAX_ARG_DEPTH:
        raise ValueError("argument nesting is too deep")
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) > MAX_STRING_LENGTH:
            raise ValueError("string argument is too long")
        return value
    if isinstance(value, list):
        if len(value) > MAX_ARG_ITEMS:
            raise ValueError("sequence argument is too long")
        return [decode_arg(item, resolve_ref, depth + 1) for item in value]
    if isinstance(value, dict) and len(value) == 1:
        (kind, inner), = value.items()
        if kind == "ref" and isinstance(inner, str):
            return resolve_ref(inner)
        if kind == "dtype" and inner in _DTYPES:
            return _DTYPES[inner]
        if kind == "device" and inner == "cpu":
            return torch.device("cpu")
        if kind == "memory_format" and inner in _MEMORY_FORMATS:
            return _MEMORY_FORMATS[inner]
        if kind == "layout" and inner == "strided":
            return torch.strided
        if kind == "float" and inner in _NONFINITE_FLOATS:
            return _NONFINITE_FLOATS[inner]
    raise ValueError("invalid argument encoding")


def _check_call_shape(call: Any) -> tuple[list[Any], dict[str, Any]]:
    if not isinstance(call, dict) or set(call) != {"args", "kwargs"}:
        raise ValueError("call must have exactly 'args' and 'kwargs'")
    args, kwargs = call["args"], call["kwargs"]
    if not isinstance(args, list) or len(args) > 64:
        raise ValueError("call args must be a short list")
    if not isinstance(kwargs, dict) or len(kwargs) > 64:
        raise ValueError("call kwargs must be a short object")
    for name in kwargs:
        if not isinstance(name, str) or not name.isidentifier() or len(name) > 64:
            raise ValueError("invalid keyword argument name")
    return args, kwargs


def call_refs(call: Any) -> list[str]:
    """Validate an encoded call and list the tensor names it reads, in order."""
    args, kwargs = _check_call_shape(call)
    refs: list[str] = []

    def collect(name: str) -> None:
        refs.append(name)

    for arg in [*args, *kwargs.values()]:
        decode_arg(arg, collect)
    return refs


# ------------------------------------------------------------------- execution
@functools.lru_cache(maxsize=None)
def _resolve_aten(op: str) -> Any:
    _, name, overload = op.split(".")
    return getattr(getattr(torch.ops.aten, name), overload)


def execute_op(op: str, call: dict[str, Any], env: dict[str, Any]) -> Any:
    """Run one allow-listed operation against already computed values."""
    if op not in ALLOWED_OPS:
        raise ValueError(f"operation not allowed: {op!r}")
    args_data, kwargs_data = _check_call_shape(call)

    def resolve(name: str) -> Any:
        try:
            return env[name]
        except KeyError:
            raise ValueError(f"undefined tensor: {name!r}") from None

    args = [decode_arg(arg, resolve) for arg in args_data]
    kwargs = {name: decode_arg(arg, resolve) for name, arg in kwargs_data.items()}

    if op == "getitem":
        if len(args) != 2 or kwargs or not isinstance(args[1], int):
            raise ValueError("getitem needs a sequence and an integer index")
        return args[0][args[1]]
    if op == "aten.dropout.default":
        training = args[2] if len(args) > 2 else kwargs.get("train", False)
        if training:
            raise ValueError("training-mode dropout is random and not allowed")
    return _resolve_aten(op)(*args, **kwargs)
