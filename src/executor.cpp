#include "dlc/executor.hpp"

#include <algorithm>
#include <stdexcept>

namespace dlc {

namespace {

// Index of element `i` of the (possibly broadcast) operand `t` when the result
// has `outSize` elements. A smaller operand is 1-D and repeats along the last dim.
inline size_t bcastIndex(size_t i, size_t tSize) { return tSize == 0 ? 0 : i % tSize; }

template <typename F>
Tensor elementwise(const Tensor& a, const Tensor& b, const std::vector<int>& outShape, F f) {
    Tensor out;
    out.shape = outShape;
    out.data.resize(numElements(outShape));
    const size_t n = out.data.size();
    const size_t as = a.size(), bs = b.size();
    for (size_t i = 0; i < n; ++i) {
        out.data[i] = f(a.data[bcastIndex(i, as)], b.data[bcastIndex(i, bs)]);
    }
    return out;
}

// C[n,m] = A[n,k] * B[k,m]. i-p-j loop order keeps B/C accesses contiguous and
// accumulates every C[i][j] in increasing p, so fused and unfused kernels
// perform exactly the same floating point operations.
void matmulInto(const float* a, const float* b, float* c, int n, int k, int m) {
    std::fill(c, c + static_cast<size_t>(n) * m, 0.0f);
    for (int i = 0; i < n; ++i) {
        float* crow = c + static_cast<size_t>(i) * m;
        for (int p = 0; p < k; ++p) {
            const float av = a[static_cast<size_t>(i) * k + p];
            const float* brow = b + static_cast<size_t>(p) * m;
            for (int j = 0; j < m; ++j) crow[j] += av * brow[j];
        }
    }
}

}  // namespace

Tensor evalNode(const Node& node, const std::vector<const Tensor*>& in) {
    switch (node.op) {
        case OpType::Add:
            return elementwise(*in[0], *in[1], node.shape, [](float x, float y) { return x + y; });
        case OpType::Mul:
            return elementwise(*in[0], *in[1], node.shape, [](float x, float y) { return x * y; });
        case OpType::ReLU: {
            Tensor out;
            out.shape = node.shape;
            out.data.resize(in[0]->size());
            for (size_t i = 0; i < out.data.size(); ++i) {
                out.data[i] = std::max(in[0]->data[i], 0.0f);
            }
            return out;
        }
        case OpType::MatMul: {
            Tensor out;
            out.shape = node.shape;
            out.data.resize(numElements(node.shape));
            matmulInto(in[0]->data.data(), in[1]->data.data(), out.data.data(), in[0]->shape[0],
                       in[0]->shape[1], in[1]->shape[1]);
            return out;
        }
        case OpType::FusedMatMulBiasReLU: {
            // One kernel, one output buffer: the matmul result is never
            // materialised as a separate tensor and the bias add + ReLU run
            // as an epilogue on the freshly computed rows.
            Tensor out;
            out.shape = node.shape;
            out.data.resize(numElements(node.shape));
            const int n = in[0]->shape[0], k = in[0]->shape[1], m = in[1]->shape[1];
            matmulInto(in[0]->data.data(), in[1]->data.data(), out.data.data(), n, k, m);
            const Tensor& bias = *in[2];
            const size_t bs = bias.size();
            for (size_t i = 0; i < out.data.size(); ++i) {
                out.data[i] = std::max(out.data[i] + bias.data[bcastIndex(i, bs)], 0.0f);
            }
            return out;
        }
        case OpType::Input:
        case OpType::Constant:
            break;
    }
    throw std::runtime_error("evalNode: not a compute node: " + node.name);
}

std::vector<Tensor> execute(const Graph& g, ExecStats* stats) {
    std::vector<Tensor> values(g.nodes.size());
    ExecStats local;
    for (const Node& n : g.nodes) {
        if (n.op == OpType::Input || n.op == OpType::Constant) {
            values[static_cast<size_t>(n.id)] = n.value;
            continue;
        }
        std::vector<const Tensor*> ins;
        for (int in : n.inputs) ins.push_back(&values[static_cast<size_t>(in)]);
        values[static_cast<size_t>(n.id)] = evalNode(n, ins);
        ++local.nodesExecuted;
        local.intermediateBytes += values[static_cast<size_t>(n.id)].bytes();
    }
    if (stats) *stats = local;

    std::vector<Tensor> outs;
    for (int o : g.outputs) outs.push_back(values[static_cast<size_t>(o)]);
    return outs;
}

}  // namespace dlc
