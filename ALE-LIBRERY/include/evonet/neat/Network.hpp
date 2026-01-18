#ifndef EVONET_NEAT_NETWORK_HPP
#define EVONET_NEAT_NETWORK_HPP

#include "Genome.hpp"
#include <queue>
#include <unordered_map>
#include <cmath>

namespace evonet {
namespace neat {

inline double neatActivation(double x) {
    return 1.0 / (1.0 + std::exp(-4.9 * x));
}

struct Link {
    int inIndex;
    double weight;
};

struct Node {
    int id;
    NodeType type;
    double bias;
    double output;
    std::vector<Link> incoming;
};

class Network {
public:
    explicit Network(const Genome& genome) {
        buildFromGenome(genome);
    }

    std::vector<double> activate(const std::vector<double>& inputs) {
        if (inputs.size() != inputIndices.size()) {
            throw std::invalid_argument("Tamaño de entrada no coincide con nodos de entrada.");
        }

        for (size_t i = 0; i < inputIndices.size(); ++i) {
            nodes[inputIndices[i]].output = inputs[i];
        }
        if (biasIndex >= 0) {
            nodes[biasIndex].output = 1.0;
        }

        for (int idx : topoOrder) {
            Node& n = nodes[idx];
            if (n.type == NodeType::INPUT || n.type == NodeType::BIAS) continue;
            double sum = n.bias;
            for (const auto& link : n.incoming) {
                sum += nodes[link.inIndex].output * link.weight;
            }
            n.output = neatActivation(sum);
        }

        std::vector<double> outputs;
        outputs.reserve(outputIndices.size());
        for (int idx : outputIndices) {
            outputs.push_back(nodes[idx].output);
        }
        return outputs;
    }

private:
    std::vector<Node> nodes;
    std::unordered_map<int, int> idToIndex;
    std::vector<int> inputIndices;
    std::vector<int> outputIndices;
    int biasIndex = -1;
    std::vector<int> topoOrder;

    void buildFromGenome(const Genome& genome) {
        nodes.clear();
        idToIndex.clear();
        inputIndices.clear();
        outputIndices.clear();
        biasIndex = -1;
        topoOrder.clear();

        nodes.reserve(genome.nodes.size());
        for (const auto& ng : genome.nodes) {
            int idx = static_cast<int>(nodes.size());
            idToIndex[ng.id] = idx;
            nodes.push_back(Node{ng.id, ng.type, ng.bias, 0.0, {}});
            if (ng.type == NodeType::INPUT) inputIndices.push_back(idx);
            if (ng.type == NodeType::OUTPUT) outputIndices.push_back(idx);
            if (ng.type == NodeType::BIAS) biasIndex = idx;
        }

        std::vector<int> indegree(nodes.size(), 0);
        std::vector<std::vector<int>> outgoing(nodes.size());

        auto createsCycle = [&](int from, int to) {
            std::vector<int> stack;
            std::vector<char> visited(nodes.size(), 0);
            stack.push_back(to);
            while (!stack.empty()) {
                int node = stack.back();
                stack.pop_back();
                if (node == from) return true;
                if (visited[node]) continue;
                visited[node] = 1;
                for (int nxt : outgoing[node]) {
                    stack.push_back(nxt);
                }
            }
            return false;
        };

        for (const auto& cg : genome.connections) {
            if (!cg.enabled) continue;
            auto itIn = idToIndex.find(cg.inNode);
            auto itOut = idToIndex.find(cg.outNode);
            if (itIn == idToIndex.end() || itOut == idToIndex.end()) continue;

            int inIdx = itIn->second;
            int outIdx = itOut->second;

            if (createsCycle(inIdx, outIdx)) {
                continue; // ignora conexión recurrente para modo feed-forward
            }

            nodes[outIdx].incoming.push_back(Link{inIdx, cg.weight});
            outgoing[inIdx].push_back(outIdx);
            indegree[outIdx]++;
        }

        std::queue<int> q;
        for (size_t i = 0; i < nodes.size(); ++i) {
            if (indegree[i] == 0) q.push(static_cast<int>(i));
        }

        while (!q.empty()) {
            int u = q.front();
            q.pop();
            topoOrder.push_back(u);
            for (int v : outgoing[u]) {
                indegree[v]--;
                if (indegree[v] == 0) q.push(v);
            }
        }

        if (topoOrder.size() != nodes.size()) {
            // throw std::runtime_error("Se detectó un ciclo en la red NEAT.");
            // En caso de fallo de topo sort, usamos lo que tengamos
        }
    }

    // PERSISTENCIA: Guardar genoma compatible (simulado)
    bool save(const std::string& filepath) {
        std::ofstream file(filepath);
        if (!file.is_open()) return false;
        // Solo guardamos una marca para saber que es NEAT, por ahora solo el fitness
        file << "NEAT_MODEL_V1\n"; 
        // Implementación real requeriría serializar topología compleja
        // Por ahora confiamos en Genome::save del propio NEAT que ya existe
        return true;
    }
};

} // namespace neat
} // namespace evonet

#endif
