#include "dlc/passes.hpp"

#include <algorithm>

#include "dlc/executor.hpp"

namespace dlc {

size_t constantFolding(Graph& g) {
    size_t folded = 0;
    for (Node& n : g.nodes) {
        if (n.op == OpType::Input || n.op == OpType::Constant) continue;

        bool allConst = !n.inputs.empty();
        std::vector<const Tensor*> ins;
        for (int in : n.inputs) {
            const Node& src = g.nodes[static_cast<size_t>(in)];
            if (src.op != OpType::Constant) {
                allConst = false;
                break;
            }
            ins.push_back(&src.value);
        }
        if (!allConst) continue;

        Tensor result = evalNode(n, ins);
        n.op = OpType::Constant;
        n.inputs.clear();
        n.value = std::move(result);
        ++folded;
    }
    return folded;
}

size_t operatorFusion(Graph& g) {
    const std::vector<int> uses = g.useCounts();
    size_t fused = 0;

    for (Node& relu : g.nodes) {
        if (relu.op != OpType::ReLU) continue;

        const Node& add = g.nodes[static_cast<size_t>(relu.inputs[0])];
        if (add.op != OpType::Add || uses[static_cast<size_t>(add.id)] != 1) continue;

        // Either operand of the add may be the matmul; the other is the bias.
        for (int side = 0; side < 2; ++side) {
            const Node& mm = g.nodes[static_cast<size_t>(add.inputs[side])];
            const int biasId = add.inputs[1 - side];
            if (mm.op != OpType::MatMul || uses[static_cast<size_t>(mm.id)] != 1) continue;
            if (mm.id == biasId) continue;  // add(mm, mm): matmul has 2 uses anyway
            // The bias must not make the add result larger than the matmul output.
            if (add.shape != mm.shape) continue;

            relu.op = OpType::FusedMatMulBiasReLU;
            relu.inputs = {mm.inputs[0], mm.inputs[1], biasId};
            ++fused;
            break;
        }
    }
    return fused;
}

size_t deadNodeElimination(Graph& g) {
    const size_t n = g.nodes.size();
    std::vector<char> live(n, 0);

    // Nodes are topologically ordered, so a single reverse sweep marks everything
    // reachable from the outputs.
    for (int o : g.outputs) live[static_cast<size_t>(o)] = 1;
    for (size_t i = n; i-- > 0;) {
        if (!live[i]) continue;
        for (int in : g.nodes[i].inputs) live[static_cast<size_t>(in)] = 1;
    }

    std::vector<int> newId(n, -1);
    std::vector<Node> kept;
    for (size_t i = 0; i < n; ++i) {
        if (!live[i]) continue;
        Node node = std::move(g.nodes[i]);
        newId[i] = static_cast<int>(kept.size());
        node.id = newId[i];
        for (int& in : node.inputs) in = newId[static_cast<size_t>(in)];
        kept.push_back(std::move(node));
    }
    for (int& o : g.outputs) o = newId[static_cast<size_t>(o)];

    const size_t removed = n - kept.size();
    g.nodes = std::move(kept);
    return removed;
}

PassStats optimize(Graph& g) {
    PassStats s;
    s.folded = constantFolding(g);
    s.fused = operatorFusion(g);
    s.eliminated = deadNodeElimination(g);
    g.validate();
    return s;
}

}  // namespace dlc
