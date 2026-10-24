"""Demo: wrap a small transformer stack and print what FSSLK captured.

    python -m fsslk.graph_demo [--layers 4] [--dim 512] [--heads 8]
                               [--batch 8] [--seq 256] [--json graph.json]
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch import nn

import fsslk


class Block(nn.Module):
    """Pre-norm transformer block with a residual stream."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.shape
        qkv = self.qkv(self.ln1(x)).view(batch, seq, 3, self.heads, dim // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(q, k, v)
        x = x + self.proj(attended.transpose(1, 2).reshape(batch, seq, dim))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


def build_model(layers: int, dim: int, heads: int) -> nn.Module:
    """The demo model: a stack of transformer blocks, in eval mode."""
    return nn.Sequential(*[Block(dim, heads) for _ in range(layers)]).eval()


def main() -> None:
    parser = argparse.ArgumentParser(prog="fsslk.graph_demo", description=__doc__)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--json", help="also write the captured graph to this file")
    args = parser.parse_args()

    torch.manual_seed(0)
    model = build_model(args.layers, args.dim, args.heads)
    example = torch.randn(args.batch, args.seq, args.dim)

    wrapped = fsslk.wrap(model, example)  # captures immediately, runs nothing remotely
    print(wrapped.graph.report())

    with torch.no_grad():
        identical = torch.equal(wrapped(example), model(example))
    print(f"\nwrapped output identical to the original: {identical}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            handle.write(wrapped.graph.to_json(indent=1))
        print(f"graph written to {args.json}")


if __name__ == "__main__":
    main()
