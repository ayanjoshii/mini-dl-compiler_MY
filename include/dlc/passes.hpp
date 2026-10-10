// Graph optimisation passes. Every pass rewrites the graph in place and keeps
// the topological-order invariant of dlc::Graph.
#pragma once

#include "dlc/graph.hpp"

namespace dlc {

struct PassStats {
    size_t folded = 0;        // compute nodes replaced by constants
    size_t fused = 0;         // matmul+add+relu chains merged into one node
    size_t eliminated = 0;    // unreachable nodes removed
};

// Replaces every compute node whose inputs are all constants by a Constant
// holding the pre-computed value. Returns the number of nodes folded.
size_t constantFolding(Graph& g);

// Rewrites relu(add(matmul(a, b), bias)) into FusedMatMulBiasReLU(a, b, bias)
// when the intermediate matmul and add results have no other consumer.
// The now-unused matmul/add nodes are left for dead-node elimination.
size_t operatorFusion(Graph& g);

// Removes every node that no graph output depends on (dead code) and
// renumbers the remaining nodes. Returns the number of nodes removed.
size_t deadNodeElimination(Graph& g);

// Default pipeline: constant folding -> operator fusion -> dead-node elimination.
PassStats optimize(Graph& g);

}  // namespace dlc
