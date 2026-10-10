#!/usr/bin/env python3
"""End-to-end correctness check against a NumPy reference.

For several random MLPs this script
  1. builds the graph text and a float32 NumPy reference result,
  2. runs the compiler binary unoptimised (--no-opt) and optimised,
  3. checks both outputs against NumPy and against each other.

Usage: verify.py [path/to/dlc]     (default: build/dlc)
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_model import build  # noqa: E402

CASES = [
    (1, [4, 4]),
    (3, [8, 16, 4]),
    (16, [32, 64, 64, 10]),
    (64, [128, 256, 256, 64]),
]


def run(binary, graph_path, *flags):
    out = subprocess.run([binary, str(graph_path), *flags], capture_output=True, text=True, check=True)
    line = out.stdout.strip().splitlines()[-1]
    values = line.split(":", 1)[1].split()
    return np.array(values, dtype=np.float32)


def main():
    root = Path(__file__).resolve().parent.parent
    binary = sys.argv[1] if len(sys.argv) > 1 else str(root / "build" / "dlc")

    failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        for seed, (batch, dims) in enumerate(CASES):
            text, expected = build(batch, dims, seed)
            path = Path(tmp) / f"case{seed}.txt"
            path.write_text(text)

            base = run(binary, path, "--no-opt")
            opt = run(binary, path)
            ref = expected.ravel()

            err_ref = float(np.max(np.abs(opt - ref)))
            err_opt = float(np.max(np.abs(opt - base)))
            ok = np.allclose(opt, ref, rtol=1e-4, atol=1e-5) and err_opt == 0.0
            failed += not ok
            print(f"batch={batch:<3} dims={dims!s:<18} |opt-numpy|={err_ref:.2e} "
                  f"|opt-baseline|={err_opt:.1e}  {'OK' if ok else 'FAIL'}")

    if failed:
        print(f"{failed} case(s) failed")
        sys.exit(1)
    print("all cases match")


if __name__ == "__main__":
    main()
