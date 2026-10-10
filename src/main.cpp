// dlc: command line driver for the mini deep learning compiler.
//
//   dlc <graph.txt> [--no-opt] [--dump-ir] [--report] [--reps N] [--quiet]
//
//   (default)   optimise the graph, run it and print the outputs
//   --no-opt    run the graph exactly as written
//   --dump-ir   print the IR before/after optimisation to stderr
//   --report    run baseline and optimised graphs, compare outputs and timings
//   --reps N    repetitions per timing measurement in --report (default 20)
//   --quiet     do not print output tensors
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include "dlc/executor.hpp"
#include "dlc/parser.hpp"
#include "dlc/passes.hpp"

using namespace dlc;

namespace {

struct Options {
    std::string path;
    bool optimize = true;
    bool dumpIr = false;
    bool report = false;
    bool quiet = false;
    int reps = 20;
};

void usage() {
    std::fprintf(stderr,
                 "usage: dlc <graph.txt> [--no-opt] [--dump-ir] [--report] [--reps N] [--quiet]\n");
}

bool parseArgs(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--no-opt") o.optimize = false;
        else if (a == "--dump-ir") o.dumpIr = true;
        else if (a == "--report") o.report = true;
        else if (a == "--quiet") o.quiet = true;
        else if (a == "--reps" && i + 1 < argc) o.reps = std::max(1, std::atoi(argv[++i]));
        else if (!a.empty() && a[0] != '-' && o.path.empty()) o.path = a;
        else return false;
    }
    return !o.path.empty();
}

double maxAbsDiff(const std::vector<Tensor>& a, const std::vector<Tensor>& b) {
    double worst = 0.0;
    for (size_t t = 0; t < a.size(); ++t) {
        for (size_t i = 0; i < a[t].data.size(); ++i) {
            worst = std::max(worst, std::fabs(static_cast<double>(a[t].data[i]) - b[t].data[i]));
        }
    }
    return worst;
}

// Mean wall-clock milliseconds of executing `g`, after one warm-up run.
double timeMs(const Graph& g, int reps) {
    execute(g);
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) execute(g);
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
}

size_t computeNodes(const Graph& g) {
    size_t c = 0;
    for (const Node& n : g.nodes) c += (n.op != OpType::Input && n.op != OpType::Constant);
    return c;
}

int runReport(const Graph& baseline, Graph optimized, int reps) {
    PassStats ps = optimize(optimized);

    ExecStats sb, so;
    std::vector<Tensor> ob = execute(baseline, &sb);
    std::vector<Tensor> oo = execute(optimized, &so);
    const double diff = maxAbsDiff(ob, oo);
    const double tb = timeMs(baseline, reps);
    const double to = timeMs(optimized, reps);

    std::printf("passes:  constant folding: %zu node(s) folded\n", ps.folded);
    std::printf("         operator fusion:  %zu chain(s) fused\n", ps.fused);
    std::printf("         dead-node elim.:  %zu node(s) removed\n", ps.eliminated);
    std::printf("\n");
    std::printf("%-22s %12s %12s\n", "", "baseline", "optimized");
    std::printf("%-22s %12zu %12zu\n", "graph nodes", baseline.nodes.size(), optimized.nodes.size());
    std::printf("%-22s %12zu %12zu\n", "compute nodes", computeNodes(baseline),
                computeNodes(optimized));
    std::printf("%-22s %12zu %12zu\n", "nodes executed", sb.nodesExecuted, so.nodesExecuted);
    std::printf("%-22s %12zu %12zu\n", "intermediate bytes", sb.intermediateBytes,
                so.intermediateBytes);
    std::printf("%-22s %12.4f %12.4f\n", "mean time (ms)", tb, to);
    std::printf("\nspeedup: %.2fx   max |baseline - optimized|: %.3g  => %s\n", tb / to, diff,
                diff <= 1e-5 ? "outputs match" : "OUTPUTS DIFFER");
    return diff <= 1e-5 ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    if (!parseArgs(argc, argv, opt)) {
        usage();
        return 1;
    }

    try {
        Graph g = parseGraphFile(opt.path);

        if (opt.report) return runReport(g, g, opt.reps);

        if (opt.dumpIr) std::fprintf(stderr, "--- IR before optimisation ---\n%s", g.dump().c_str());
        if (opt.optimize) {
            PassStats ps = optimize(g);
            if (opt.dumpIr) {
                std::fprintf(stderr,
                             "--- IR after optimisation (folded=%zu fused=%zu removed=%zu) ---\n%s",
                             ps.folded, ps.fused, ps.eliminated, g.dump().c_str());
            }
        }
        std::vector<Tensor> outs = execute(g);
        if (!opt.quiet) std::fputs(formatOutputs(g, outs).c_str(), stdout);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
