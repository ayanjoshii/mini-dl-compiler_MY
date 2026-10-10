// Reference CPU executor and the kernels it uses.
#pragma once

#include <vector>

#include "dlc/graph.hpp"

namespace dlc {

struct ExecStats {
    size_t nodesExecuted = 0;        // nodes that actually ran (inputs/consts excluded)
    size_t intermediateBytes = 0;    // bytes of tensors produced by executed nodes
};

// Evaluates a single compute node given pointers to its input tensors.
Tensor evalNode(const Node& node, const std::vector<const Tensor*>& inputs);

// Runs every node of the graph in order (including nodes that do not feed any
// output) and returns the tensors of the graph outputs.
std::vector<Tensor> execute(const Graph& g, ExecStats* stats = nullptr);

}  // namespace dlc
