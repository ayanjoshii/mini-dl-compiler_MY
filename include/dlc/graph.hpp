// Computational-graph intermediate representation (IR).
//
// A Graph is a list of nodes stored in topological order: every node only
// reads from nodes with a smaller id. All passes preserve this invariant, so
// the executor can simply walk the node list front to back.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace dlc {

enum class OpType {
    Input,                 // runtime-fed tensor (value holds the feed)
    Constant,              // compile-time known tensor
    Add,                   // element-wise add, supports 1-D (last-dim) broadcast
    Mul,                   // element-wise multiply, same broadcast rule
    MatMul,                // [n,k] x [k,m] -> [n,m]
    ReLU,                  // max(x, 0)
    FusedMatMulBiasReLU,   // relu(matmul(a, b) + bias) in a single kernel
};

const char* opName(OpType op);

struct Tensor {
    std::vector<int> shape;
    std::vector<float> data;

    size_t size() const { return data.size(); }
    size_t bytes() const { return data.size() * sizeof(float); }
};

size_t numElements(const std::vector<int>& shape);
std::string shapeToString(const std::vector<int>& shape);

struct Node {
    int id = -1;
    OpType op = OpType::Constant;
    std::string name;
    std::vector<int> inputs;
    Tensor value;            // only meaningful for Input / Constant
    std::vector<int> shape;  // inferred output shape
};

class Graph {
public:
    std::vector<Node> nodes;
    std::vector<int> outputs;

    // Adds a node and infers its output shape. Throws std::runtime_error when
    // the inputs are invalid (unknown id, shape mismatch, wrong arity).
    int addNode(OpType op, const std::string& name, const std::vector<int>& inputs,
                Tensor value = Tensor{});

    // Number of consumers of every node (graph outputs count as a use).
    std::vector<int> useCounts() const;

    // Checks the topological-order invariant and shape consistency.
    void validate() const;

    // Human readable IR listing, one node per line.
    std::string dump() const;
};

// Output shape of `op` given its input shapes; throws on mismatch.
std::vector<int> inferShape(OpType op, const std::vector<std::vector<int>>& inShapes);

}  // namespace dlc
