# Mini Deep Learning Compiler & Graph Optimizer

A small compiler framework in C++ that represents neural-network computations as a
**computational graph** in a custom **intermediate representation (IR)**, rewrites that
graph with **optimization passes**, and executes it on a CPU reference runtime.
A Python harness checks every result against NumPy.

```
 text graph ──► parser ──► Graph IR ──► passes ──► optimized IR ──► executor ──► outputs
                                         │
                      constant folding · operator fusion · dead-node elimination
```

## Features

| Pass | What it does |
|---|---|
| **Constant folding** | Any node whose inputs are all constants is evaluated at compile time and replaced by a `const`. |
| **Operator fusion** | `relu(add(matmul(a, b), bias))` becomes one `fused_matmul_bias_relu` node with a single kernel and a single output buffer, provided the intermediate results have no other consumer. |
| **Dead-node elimination** | Nodes that no graph output depends on are removed and the graph is renumbered. |

Supporting pieces: shape inference and graph validation, a line-based text front-end,
reference kernels (`add`, `mul`, `matmul`, `relu`, fused kernel), and an execution report
that compares baseline vs. optimized graphs for correctness and performance.

## Build

Requires a C++17 compiler and CMake. Python 3 + NumPy are only needed for the verification scripts.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure      # C++ unit tests
pip install -r python/requirements.txt
python3 python/verify.py                        # end-to-end check against NumPy
```

## Usage

```bash
./build/dlc examples/mlp_small.txt                  # optimize, run, print outputs
./build/dlc examples/mlp_small.txt --no-opt         # run the graph exactly as written
./build/dlc examples/mlp_small.txt --dump-ir --quiet   # IR before / after the passes
./build/dlc examples/mlp_small.txt --report         # baseline vs optimized comparison
```

Generate a bigger model and benchmark it:

```bash
python3 python/gen_model.py examples/mlp_large.txt --batch 128 --dims 256 512 512 128
./build/dlc examples/mlp_large.txt --report --reps 30
```

### Graph text format

```
input  x  4 8 : <32 values>        # runtime feed, shape [4,8]
const  w1 8 16 : <128 values>      # compile-time constant
const  b1 16 : <16 values>
matmul mm x w1
add    pre mm b1                   # 1-D operand broadcasts along the last dim
relu   h pre
output h
```

Supported ops: `input`, `const`, `add`, `mul`, `matmul`, `relu`. Nodes must be defined before use,
which keeps the graph in topological order.

### IR before and after optimization

```
--- IR before optimisation ---          --- IR after optimisation ---
%4 = add %2 %3        # bias1            %2 = const            # bias1 (folded)
%5 = matmul %0 %1     # mm1              %3 = fused_matmul_bias_relu %0 %1 %2
%6 = add %5 %4        # pre1
%7 = relu %6          # h1
%8 = mul %7 %7        # dead1            (removed)
```

## Results

`--report` runs the unoptimized and optimized graphs, then prints node counts, bytes of
intermediate tensors, mean runtime and the maximum absolute difference between outputs.
Example run (2-vCPU Linux VM, `-O2`, 2-layer MLP; your numbers will differ):

| Model | Nodes | Executed nodes | Intermediate bytes | Time | Max output diff |
|---|---|---|---|---|---|
| `mlp_small` (4x8 -> 16 -> 4) | 17 -> 7 | 10 -> 2 | 1,360 -> 320 | 2.3x faster | 0 |
| `mlp_large` (128x256 -> 512 -> 512 -> 128) | 25 -> 10 | 15 -> 3 | 2.36 MB -> 0.59 MB | 1.06x faster | 0 |

The gain shrinks as matrices grow because the naive matmul dominates the runtime; the
fusion and dead-code savings are most visible in memory traffic and on small graphs.
Fused and unfused kernels accumulate in the same order, so optimized outputs are
bit-identical to the baseline and match NumPy to about 1e-7.

## Project layout

```
include/dlc/   graph.hpp  parser.hpp  executor.hpp  passes.hpp
src/           graph.cpp  parser.cpp  executor.cpp  passes.cpp  main.cpp (CLI)
tests/         test_dlc.cpp          unit tests (shape inference, each pass, parser, pipeline)
python/        gen_model.py          random MLP generator with foldable/fusable/dead nodes
               verify.py             NumPy reference check for baseline and optimized runs
examples/      mlp_small.txt
```

## Limitations and ideas

- CPU only, `float32`, 2-D matmul; no convolutions, no autodiff.
- The matmul kernel is a simple loop nest (no tiling or SIMD), so absolute speed is modest.
- Possible next steps: algebraic simplification (`x*1`, `x+0`), common-subexpression elimination,
  more fusion patterns (conv + bias + activation), cache-blocked matmul, a CUDA backend.

## License

MIT
