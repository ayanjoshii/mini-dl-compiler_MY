// Dependency-free unit tests. Run with `ctest` or `./build/test_dlc`.
#include <cmath>
#include <cstdio>
#include <random>
#include <sstream>
#include <stdexcept>

#include "dlc/executor.hpp"
#include "dlc/parser.hpp"
#include "dlc/passes.hpp"

using namespace dlc;

static int g_failed = 0;
static int g_checks = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        ++g_checks;                                                              \
        if (!(cond)) {                                                           \
            ++g_failed;                                                          \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);          \
        }                                                                        \
    } while (0)

template <typename F>
static bool throws(F f) {
    try {
        f();
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

static Tensor randomTensor(std::vector<int> shape, std::mt19937& rng) {
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    Tensor t;
    t.shape = shape;
    t.data.resize(numElements(shape));
    for (float& v : t.data) v = d(rng);
    return t;
}

static size_t count(const Graph& g, OpType op) {
    size_t c = 0;
    for (const Node& n : g.nodes) c += (n.op == op);
    return c;
}

static double maxDiff(const std::vector<Tensor>& a, const std::vector<Tensor>& b) {
    double w = 0;
    for (size_t t = 0; t < a.size(); ++t)
        for (size_t i = 0; i < a[t].data.size(); ++i)
            w = std::fmax(w, std::fabs(double(a[t].data[i]) - b[t].data[i]));
    return w;
}

// y = relu(x @ w + (b1 + b2)) plus an unused dead branch.
static Graph buildMlpLayer(std::mt19937& rng) {
    Graph g;
    int x = g.addNode(OpType::Input, "x", {}, randomTensor({4, 8}, rng));
    int w = g.addNode(OpType::Constant, "w", {}, randomTensor({8, 6}, rng));
    int b1 = g.addNode(OpType::Constant, "b1", {}, randomTensor({6}, rng));
    int b2 = g.addNode(OpType::Constant, "b2", {}, randomTensor({6}, rng));
    int bias = g.addNode(OpType::Add, "bias", {b1, b2});        // foldable
    int mm = g.addNode(OpType::MatMul, "mm", {x, w});
    int add = g.addNode(OpType::Add, "add", {mm, bias});
    int y = g.addNode(OpType::ReLU, "y", {add});
    g.addNode(OpType::Mul, "dead", {x, x});                      // unused
    g.outputs = {y};
    g.validate();
    return g;
}

static void testShapeInference() {
    CHECK(inferShape(OpType::MatMul, {{2, 3}, {3, 5}}) == (std::vector<int>{2, 5}));
    CHECK(inferShape(OpType::Add, {{4, 6}, {6}}) == (std::vector<int>{4, 6}));
    CHECK(throws([] { inferShape(OpType::MatMul, {{2, 3}, {4, 5}}); }));
    CHECK(throws([] { inferShape(OpType::Add, {{4, 6}, {5}}); }));
}

static void testConstantFolding() {
    std::mt19937 rng(1);
    Graph g = buildMlpLayer(rng);
    CHECK(count(g, OpType::Add) == 2);
    size_t folded = constantFolding(g);
    CHECK(folded == 1);  // only b1 + b2
    const Node& bias = g.nodes[4];
    CHECK(bias.op == OpType::Constant);
    CHECK(bias.inputs.empty());
    // folded value equals the manual sum
    for (size_t i = 0; i < 6; ++i) {
        CHECK(bias.value.data[i] == g.nodes[2].value.data[i] + g.nodes[3].value.data[i]);
    }
    CHECK(constantFolding(g) == 0);  // idempotent
}

static void testDeadNodeElimination() {
    std::mt19937 rng(2);
    Graph g = buildMlpLayer(rng);
    const size_t before = g.nodes.size();
    size_t removed = deadNodeElimination(g);
    CHECK(removed == 1);  // the unused mul
    CHECK(g.nodes.size() == before - 1);
    g.validate();
    CHECK(deadNodeElimination(g) == 0);
}

static void testFusion() {
    std::mt19937 rng(3);
    Graph g = buildMlpLayer(rng);
    CHECK(operatorFusion(g) == 1);
    deadNodeElimination(g);
    g.validate();
    CHECK(count(g, OpType::FusedMatMulBiasReLU) == 1);
    CHECK(count(g, OpType::MatMul) == 0);
    CHECK(count(g, OpType::ReLU) == 0);
}

static void testFusionRespectsOtherUses() {
    std::mt19937 rng(4);
    Graph g;
    int x = g.addNode(OpType::Input, "x", {}, randomTensor({2, 3}, rng));
    int w = g.addNode(OpType::Constant, "w", {}, randomTensor({3, 3}, rng));
    int b = g.addNode(OpType::Constant, "b", {}, randomTensor({3}, rng));
    int mm = g.addNode(OpType::MatMul, "mm", {x, w});
    int add = g.addNode(OpType::Add, "add", {mm, b});
    int y = g.addNode(OpType::ReLU, "y", {add});
    // `mm` is also an output, so it must stay materialised.
    g.outputs = {y, mm};
    CHECK(operatorFusion(g) == 0);

    Graph h = g;
    h.outputs = {y, add};  // `add` has a second use
    CHECK(operatorFusion(h) == 0);
}

static void testPipelinePreservesResults() {
    std::mt19937 rng(5);
    for (int trial = 0; trial < 20; ++trial) {
        Graph base;
        int x = base.addNode(OpType::Input, "x", {}, randomTensor({7, 16}, rng));
        int cur = x;
        int width = 16;
        for (int layer = 0; layer < 4; ++layer) {
            int out = 8 + 4 * layer;
            int w = base.addNode(OpType::Constant, "w", {}, randomTensor({width, out}, rng));
            int b1 = base.addNode(OpType::Constant, "b1", {}, randomTensor({out}, rng));
            int b2 = base.addNode(OpType::Constant, "b2", {}, randomTensor({out}, rng));
            int bias = base.addNode(OpType::Add, "bias", {b1, b2});
            int mm = base.addNode(OpType::MatMul, "mm", {cur, w});
            // alternate operand order to exercise both fusion patterns
            int add = (layer % 2) ? base.addNode(OpType::Add, "add", {bias, mm})
                                  : base.addNode(OpType::Add, "add", {mm, bias});
            cur = base.addNode(OpType::ReLU, "relu", {add});
            base.addNode(OpType::Mul, "dead", {cur, cur});
            width = out;
        }
        base.outputs = {cur};

        Graph opt = base;
        PassStats ps = optimize(opt);
        CHECK(ps.fused == 4);
        CHECK(ps.folded == 4);
        CHECK(opt.nodes.size() < base.nodes.size());

        ExecStats sb, so;
        auto ob = execute(base, &sb);
        auto oo = execute(opt, &so);
        CHECK(maxDiff(ob, oo) == 0.0);  // bit-identical
        CHECK(so.nodesExecuted < sb.nodesExecuted);
        CHECK(so.intermediateBytes < sb.intermediateBytes);
    }
}

static void testParser() {
    std::istringstream ok(
        "# tiny\n"
        "input x 1 2 : 1 -2\n"
        "const w 2 2 : 1 0 0 1\n"
        "matmul y x w\n"
        "relu z y\n"
        "output z\n");
    Graph g = parseGraph(ok);
    auto outs = execute(g);
    CHECK(outs.size() == 1);
    CHECK(outs[0].data.size() == 2);
    CHECK(outs[0].data[0] == 1.0f && outs[0].data[1] == 0.0f);

    std::istringstream unknown("input x 1 : 1\nrelu y nope\noutput y\n");
    CHECK(throws([&] { parseGraph(unknown); }));
    std::istringstream badCount("input x 2 2 : 1 2 3\noutput x\n");
    CHECK(throws([&] { parseGraph(badCount); }));
    std::istringstream noOutput("input x 1 : 1\n");
    CHECK(throws([&] { parseGraph(noOutput); }));
    std::istringstream badShape("input a 2 3 : 1 2 3 4 5 6\ninput b 2 3 : 1 2 3 4 5 6\n"
                                "matmul c a b\noutput c\n");
    CHECK(throws([&] { parseGraph(badShape); }));
}

int main() {
    testShapeInference();
    testConstantFolding();
    testDeadNodeElimination();
    testFusion();
    testFusionRespectsOtherUses();
    testPipelinePreservesResults();
    testParser();
    std::printf("%d checks, %d failed\n", g_checks, g_failed);
    return g_failed == 0 ? 0 : 1;
}
