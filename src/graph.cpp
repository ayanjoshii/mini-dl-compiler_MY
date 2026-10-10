#include "dlc/graph.hpp"

#include <sstream>
#include <stdexcept>

namespace dlc {

const char* opName(OpType op) {
    switch (op) {
        case OpType::Input: return "input";
        case OpType::Constant: return "const";
        case OpType::Add: return "add";
        case OpType::Mul: return "mul";
        case OpType::MatMul: return "matmul";
        case OpType::ReLU: return "relu";
        case OpType::FusedMatMulBiasReLU: return "fused_matmul_bias_relu";
    }
    return "?";
}

size_t numElements(const std::vector<int>& shape) {
    size_t n = 1;
    for (int d : shape) n *= static_cast<size_t>(d);
    return n;
}

std::string shapeToString(const std::vector<int>& shape) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) os << ",";
        os << shape[i];
    }
    os << "]";
    return os.str();
}

namespace {

// Element-wise broadcast rule used by Add / Mul / the fused bias:
// shapes are equal, or one side is 1-D and matches the last dim of the other.
// Returns the larger shape.
std::vector<int> broadcastShape(const std::vector<int>& a, const std::vector<int>& b,
                                const char* what) {
    if (a == b) return a;
    if (b.size() == 1 && !a.empty() && a.back() == b[0]) return a;
    if (a.size() == 1 && !b.empty() && b.back() == a[0]) return b;
    throw std::runtime_error(std::string(what) + ": cannot broadcast " + shapeToString(a) +
                             " with " + shapeToString(b));
}

void requireArity(OpType op, size_t got, size_t want) {
    if (got != want) {
        throw std::runtime_error(std::string(opName(op)) + " expects " + std::to_string(want) +
                                 " input(s), got " + std::to_string(got));
    }
}

}  // namespace

std::vector<int> inferShape(OpType op, const std::vector<std::vector<int>>& in) {
    switch (op) {
        case OpType::Input:
        case OpType::Constant:
            throw std::runtime_error("inferShape: Input/Constant shapes come from their value");
        case OpType::Add:
        case OpType::Mul:
            requireArity(op, in.size(), 2);
            return broadcastShape(in[0], in[1], opName(op));
        case OpType::ReLU:
            requireArity(op, in.size(), 1);
            return in[0];
        case OpType::MatMul: {
            requireArity(op, in.size(), 2);
            if (in[0].size() != 2 || in[1].size() != 2 || in[0][1] != in[1][0]) {
                throw std::runtime_error("matmul: incompatible shapes " + shapeToString(in[0]) +
                                         " x " + shapeToString(in[1]));
            }
            return {in[0][0], in[1][1]};
        }
        case OpType::FusedMatMulBiasReLU: {
            requireArity(op, in.size(), 3);
            std::vector<int> mm = inferShape(OpType::MatMul, {in[0], in[1]});
            if (broadcastShape(mm, in[2], "fused bias") != mm) {
                throw std::runtime_error("fused_matmul_bias_relu: bias larger than matmul output");
            }
            return mm;
        }
    }
    throw std::runtime_error("inferShape: unknown op");
}

int Graph::addNode(OpType op, const std::string& name, const std::vector<int>& inputs,
                   Tensor value) {
    Node n;
    n.id = static_cast<int>(nodes.size());
    n.op = op;
    n.name = name;
    n.inputs = inputs;

    if (op == OpType::Input || op == OpType::Constant) {
        if (!inputs.empty()) throw std::runtime_error(name + ": input/const take no inputs");
        if (numElements(value.shape) != value.data.size()) {
            throw std::runtime_error(name + ": value size does not match its shape");
        }
        n.shape = value.shape;
        n.value = std::move(value);
    } else {
        std::vector<std::vector<int>> shapes;
        for (int in : inputs) {
            if (in < 0 || in >= n.id) throw std::runtime_error(name + ": invalid input id");
            shapes.push_back(nodes[static_cast<size_t>(in)].shape);
        }
        n.shape = inferShape(op, shapes);
    }
    nodes.push_back(std::move(n));
    return nodes.back().id;
}

std::vector<int> Graph::useCounts() const {
    std::vector<int> uses(nodes.size(), 0);
    for (const Node& n : nodes)
        for (int in : n.inputs) ++uses[static_cast<size_t>(in)];
    for (int o : outputs) ++uses[static_cast<size_t>(o)];
    return uses;
}

void Graph::validate() const {
    for (size_t i = 0; i < nodes.size(); ++i) {
        const Node& n = nodes[i];
        if (n.id != static_cast<int>(i)) throw std::runtime_error("node id/index mismatch");
        std::vector<std::vector<int>> shapes;
        for (int in : n.inputs) {
            if (in < 0 || in >= n.id) {
                throw std::runtime_error(n.name + ": violates topological order");
            }
            shapes.push_back(nodes[static_cast<size_t>(in)].shape);
        }
        if (n.op != OpType::Input && n.op != OpType::Constant) {
            if (inferShape(n.op, shapes) != n.shape) {
                throw std::runtime_error(n.name + ": stored shape disagrees with inference");
            }
        }
    }
    for (int o : outputs) {
        if (o < 0 || o >= static_cast<int>(nodes.size())) {
            throw std::runtime_error("output refers to unknown node");
        }
    }
}

std::string Graph::dump() const {
    std::ostringstream os;
    for (const Node& n : nodes) {
        os << "%" << n.id << " = " << opName(n.op);
        for (int in : n.inputs) os << " %" << in;
        os << "  : " << shapeToString(n.shape) << "  # " << n.name << "\n";
    }
    os << "outputs:";
    for (int o : outputs) os << " %" << o;
    os << "\n";
    return os.str();
}

}  // namespace dlc
