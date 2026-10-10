#include "dlc/parser.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace dlc {

namespace {

[[noreturn]] void fail(int line, const std::string& msg) {
    throw std::runtime_error("line " + std::to_string(line) + ": " + msg);
}

// Parses "d0 d1 ... : v0 v1 ..." into a tensor.
Tensor parseTensor(std::istringstream& ss, int line) {
    Tensor t;
    std::string tok;
    bool sawColon = false;
    while (ss >> tok) {
        if (tok == ":") {
            sawColon = true;
            break;
        }
        try {
            t.shape.push_back(std::stoi(tok));
        } catch (...) {
            fail(line, "bad dimension '" + tok + "'");
        }
    }
    if (!sawColon) fail(line, "expected ':' between shape and values");
    if (t.shape.empty()) fail(line, "tensor needs at least one dimension");
    while (ss >> tok) {
        try {
            t.data.push_back(std::stof(tok));
        } catch (...) {
            fail(line, "bad value '" + tok + "'");
        }
    }
    if (t.data.size() != numElements(t.shape)) {
        fail(line, "expected " + std::to_string(numElements(t.shape)) + " values, got " +
                       std::to_string(t.data.size()));
    }
    return t;
}

}  // namespace

Graph parseGraph(std::istream& in) {
    Graph g;
    std::unordered_map<std::string, int> ids;
    std::string raw;
    int lineNo = 0;

    auto lookup = [&](const std::string& name, int line) {
        auto it = ids.find(name);
        if (it == ids.end()) fail(line, "unknown node '" + name + "'");
        return it->second;
    };

    while (std::getline(in, raw)) {
        ++lineNo;
        size_t hash = raw.find('#');
        if (hash != std::string::npos) raw.resize(hash);
        std::istringstream ss(raw);
        std::string op, name;
        if (!(ss >> op)) continue;  // blank line

        try {
            if (op == "output") {
                if (!(ss >> name)) fail(lineNo, "output needs a node name");
                g.outputs.push_back(lookup(name, lineNo));
                continue;
            }
            if (!(ss >> name)) fail(lineNo, op + " needs a node name");
            if (ids.count(name)) fail(lineNo, "node '" + name + "' defined twice");

            int id;
            if (op == "input" || op == "const") {
                Tensor t = parseTensor(ss, lineNo);
                id = g.addNode(op == "input" ? OpType::Input : OpType::Constant, name, {}, t);
            } else {
                OpType type;
                size_t arity;
                if (op == "add") { type = OpType::Add; arity = 2; }
                else if (op == "mul") { type = OpType::Mul; arity = 2; }
                else if (op == "matmul") { type = OpType::MatMul; arity = 2; }
                else if (op == "relu") { type = OpType::ReLU; arity = 1; }
                else fail(lineNo, "unknown op '" + op + "'");

                std::vector<int> args;
                std::string a;
                while (ss >> a) args.push_back(lookup(a, lineNo));
                if (args.size() != arity) {
                    fail(lineNo, op + " expects " + std::to_string(arity) + " operand(s)");
                }
                id = g.addNode(type, name, args);
            }
            ids[name] = id;
        } catch (const std::runtime_error& e) {
            std::string msg = e.what();
            if (msg.rfind("line ", 0) == 0) throw;  // already carries a line number
            fail(lineNo, msg);
        }
    }
    if (g.outputs.empty()) throw std::runtime_error("graph has no 'output' statement");
    g.validate();
    return g;
}

Graph parseGraphFile(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    return parseGraph(f);
}

std::string formatOutputs(const Graph& g, const std::vector<Tensor>& outs) {
    std::ostringstream os;
    for (size_t i = 0; i < outs.size(); ++i) {
        os << "output " << g.nodes[static_cast<size_t>(g.outputs[i])].name
           << " shape=" << shapeToString(outs[i].shape) << " :";
        char buf[32];
        for (float v : outs[i].data) {
            std::snprintf(buf, sizeof buf, " %.9g", v);
            os << buf;
        }
        os << "\n";
    }
    return os.str();
}

}  // namespace dlc
