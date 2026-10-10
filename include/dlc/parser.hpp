// Text front-end: builds a Graph from a small line-based description.
//
//   input  NAME d0 [d1 ...] : v0 v1 ...     runtime feed (values given inline)
//   const  NAME d0 [d1 ...] : v0 v1 ...     compile-time constant
//   add|mul|matmul NAME A B
//   relu   NAME A
//   output NAME
//
// '#' starts a comment. Nodes must be defined before use.
#pragma once

#include <istream>
#include <string>

#include "dlc/graph.hpp"

namespace dlc {

Graph parseGraph(std::istream& in);
Graph parseGraphFile(const std::string& path);

// Writes the graph outputs as text: "output NAME shape=[..] : v0 v1 ..."
std::string formatOutputs(const Graph& g, const std::vector<Tensor>& outs);

}  // namespace dlc
