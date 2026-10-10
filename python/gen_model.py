#!/usr/bin/env python3
"""Generate an MLP graph in the dlc text format.

The generated graph deliberately contains the three things the compiler
optimises:
  * constant sub-expressions   (bias = b1 + b2, both constants)
  * matmul -> add -> relu      (fusion candidates)
  * dead branches              (nodes that no output depends on)

Usage: gen_model.py OUT.txt [--batch 64] [--dims 128 256 256 64] [--seed 0]
"""
import argparse

import numpy as np


def fmt(a):
    return " ".join(f"{v:.9g}" for v in a.ravel())


def build(batch, dims, seed):
    """Returns (graph_text, numpy_reference_output)."""
    rng = np.random.default_rng(seed)
    lines = []

    def tensor(kind, name, arr):
        shape = " ".join(str(d) for d in arr.shape)
        lines.append(f"{kind} {name} {shape} : {fmt(arr)}")

    x = rng.uniform(-1, 1, size=(batch, dims[0])).astype(np.float32)
    tensor("input", "x", x)

    cur, cur_name = x, "x"
    for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:]), start=1):
        scale = np.float32(1.0 / np.sqrt(d_in))
        w = (rng.uniform(-1, 1, size=(d_in, d_out)) * scale).astype(np.float32)
        b1 = rng.uniform(-0.1, 0.1, size=(d_out,)).astype(np.float32)
        b2 = rng.uniform(-0.1, 0.1, size=(d_out,)).astype(np.float32)
        tensor("const", f"w{i}", w)
        tensor("const", f"b{i}a", b1)
        tensor("const", f"b{i}b", b2)
        lines.append(f"add bias{i} b{i}a b{i}b")           # constant folding target
        lines.append(f"matmul mm{i} {cur_name} w{i}")
        lines.append(f"add pre{i} mm{i} bias{i}")
        lines.append(f"relu h{i} pre{i}")
        lines.append(f"mul dead{i} h{i} h{i}")             # dead code: nobody reads it

        cur = np.maximum(cur @ w + (b1 + b2), 0).astype(np.float32)
        cur_name = f"h{i}"

    lines.append(f"output {cur_name}")
    return "\n".join(lines) + "\n", cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dims", type=int, nargs="+", default=[128, 256, 256, 64])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    text, _ = build(args.batch, args.dims, args.seed)
    with open(args.out, "w") as f:
        f.write(text)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
