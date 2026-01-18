#ifndef EVONET_NEAT_GENOME_HPP
#define EVONET_NEAT_GENOME_HPP

#include "NeatConfig.hpp"
#include "InnovationDatabase.hpp"

#include <algorithm>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <stdexcept>
#include <cmath>

#include <fstream>
#include <iomanip>

namespace evonet {
namespace neat {

enum class NodeType {
    INPUT,
    BIAS,
    HIDDEN,
    OUTPUT
};

struct NodeGene {
    int id;
    NodeType type;
    double bias;
};

struct ConnectionGene {
    int inNode;
    int outNode;
    double weight;
    bool enabled;
    int innovation;

    bool operator<(const ConnectionGene& other) const {
        return innovation < other.innovation;
    }
};

class Genome {
public:
    double fitness = 0.0;
    double adjustedFitness = 0.0;

    std::vector<NodeGene> nodes;
    std::vector<ConnectionGene> connections;

    Genome() = default;

    Genome(const std::vector<int>& inputIds,
           int biasId,
           const std::vector<int>& outputIds,
           InnovationDatabase& innovDb,
           std::mt19937& rng) {
        nodes.clear();
        connections.clear();

        for (int id : inputIds) {
            nodes.push_back(NodeGene{id, NodeType::INPUT, 0.0});
        }
        nodes.push_back(NodeGene{biasId, NodeType::BIAS, 0.0});
        for (int id : outputIds) {
            nodes.push_back(NodeGene{id, NodeType::OUTPUT, 0.0});
        }

        std::uniform_real_distribution<double> wdist(-1.0, 1.0);
        for (int inId : inputIds) {
            for (int outId : outputIds) {
                connections.push_back(ConnectionGene{
                    inId,
                    outId,
                    wdist(rng),
                    true,
                    innovDb.getInnovation(inId, outId)
                });
            }
        }
        for (int outId : outputIds) {
            connections.push_back(ConnectionGene{
                biasId,
                outId,
                wdist(rng),
                true,
                innovDb.getInnovation(biasId, outId)
            });
        }

        std::sort(connections.begin(), connections.end());
    }

    void mutateWeights(const NeatConfig& cfg, std::mt19937& rng) {
        std::normal_distribution<double> perturb(0.0, cfg.weightPerturbStd);
        std::uniform_real_distribution<double> chance(0.0, 1.0);
        std::uniform_real_distribution<double> replace(-cfg.weightReplaceRange, cfg.weightReplaceRange);

        for (auto& c : connections) {
            if (chance(rng) < cfg.weightMutationRate) {
                if (chance(rng) < cfg.weightPerturbRate) {
                    c.weight += perturb(rng);
                } else {
                    c.weight = replace(rng);
                }
            }
        }
    }

    void mutateAddConnection(const NeatConfig& cfg,
                             InnovationDatabase& innovDb,
                             std::mt19937& rng) {
        std::uniform_int_distribution<size_t> nodeDist(0, nodes.size() - 1);
        std::uniform_real_distribution<double> wdist(-1.0, 1.0);

        for (size_t attempt = 0; attempt < cfg.addConnectionTries; ++attempt) {
            const NodeGene& a = nodes[nodeDist(rng)];
            const NodeGene& b = nodes[nodeDist(rng)];

            if (a.id == b.id) continue;
            if (b.type == NodeType::INPUT || b.type == NodeType::BIAS) continue;
            if (a.type == NodeType::OUTPUT && b.type == NodeType::OUTPUT) continue;

            if (hasConnection(a.id, b.id)) continue;
            if (createsCycle(a.id, b.id)) continue;

            connections.push_back(ConnectionGene{
                a.id,
                b.id,
                wdist(rng),
                true,
                innovDb.getInnovation(a.id, b.id)
            });
            std::sort(connections.begin(), connections.end());
            return;
        }
    }

    void mutateAddNode(InnovationDatabase& innovDb, std::mt19937& rng) {
        std::vector<size_t> enabledIdx;
        enabledIdx.reserve(connections.size());
        for (size_t i = 0; i < connections.size(); ++i) {
            if (connections[i].enabled) enabledIdx.push_back(i);
        }
        if (enabledIdx.empty()) return;

        std::uniform_int_distribution<size_t> pick(0, enabledIdx.size() - 1);
        size_t idx = enabledIdx[pick(rng)];
        ConnectionGene& target = connections[idx];

        target.enabled = false;
        int newNodeId = innovDb.createNodeId();
        nodes.push_back(NodeGene{newNodeId, NodeType::HIDDEN, 0.0});

        connections.push_back(ConnectionGene{
            target.inNode,
            newNodeId,
            1.0,
            true,
            innovDb.getInnovation(target.inNode, newNodeId)
        });
        connections.push_back(ConnectionGene{
            newNodeId,
            target.outNode,
            target.weight,
            true,
            innovDb.getInnovation(newNodeId, target.outNode)
        });

        std::sort(connections.begin(), connections.end());
    }

    static Genome crossover(const Genome& fitParent, const Genome& otherParent, std::mt19937& rng) {
        Genome child;
        child.fitness = 0.0;
        child.adjustedFitness = 0.0;

        std::unordered_map<int, NodeGene> nodeMap;
        for (const auto& n : fitParent.nodes) {
            nodeMap[n.id] = n;
        }
        for (const auto& n : otherParent.nodes) {
            if (nodeMap.find(n.id) == nodeMap.end()) {
                nodeMap[n.id] = n;
            }
        }

        std::uniform_int_distribution<int> coin(0, 1);
        size_t i = 0;
        size_t j = 0;
        while (i < fitParent.connections.size() || j < otherParent.connections.size()) {
            if (i >= fitParent.connections.size()) {
                break;
            }
            if (j >= otherParent.connections.size()) {
                child.connections.push_back(fitParent.connections[i++]);
                continue;
            }

            const auto& g1 = fitParent.connections[i];
            const auto& g2 = otherParent.connections[j];

            if (g1.innovation == g2.innovation) {
                child.connections.push_back(coin(rng) ? g1 : g2);
                ++i;
                ++j;
            } else if (g1.innovation < g2.innovation) {
                child.connections.push_back(g1);
                ++i;
            } else {
                ++j;
            }
        }

        std::unordered_set<int> usedNodes;
        for (const auto& c : child.connections) {
            usedNodes.insert(c.inNode);
            usedNodes.insert(c.outNode);
        }
        for (const auto& kv : nodeMap) {
            if (usedNodes.count(kv.first) > 0) {
                child.nodes.push_back(kv.second);
            }
        }

        std::sort(child.connections.begin(), child.connections.end());
        return child;
    }

    double compatibilityDistance(const Genome& other, const NeatConfig& cfg) const {
        size_t i = 0;
        size_t j = 0;
        int excess = 0;
        int disjoint = 0;
        double weightDiff = 0.0;
        int matching = 0;

        int maxInnov1 = connections.empty() ? 0 : connections.back().innovation;
        int maxInnov2 = other.connections.empty() ? 0 : other.connections.back().innovation;

        while (i < connections.size() && j < other.connections.size()) {
            const auto& g1 = connections[i];
            const auto& g2 = other.connections[j];

            if (g1.innovation == g2.innovation) {
                matching++;
                weightDiff += std::abs(g1.weight - g2.weight);
                ++i;
                ++j;
            } else if (g1.innovation < g2.innovation) {
                if (g1.innovation > maxInnov2) excess++; else disjoint++;
                ++i;
            } else {
                if (g2.innovation > maxInnov1) excess++; else disjoint++;
                ++j;
            }
        }

        for (; i < connections.size(); ++i) {
            if (connections[i].innovation > maxInnov2) excess++; else disjoint++;
        }
        for (; j < other.connections.size(); ++j) {
            if (other.connections[j].innovation > maxInnov1) excess++; else disjoint++;
        }

        double N = static_cast<double>(std::max(connections.size(), other.connections.size()));
        if (N < 20.0) N = 1.0;
        double W = (matching == 0) ? 0.0 : weightDiff / matching;

        return (cfg.c1 * excess) / N + (cfg.c2 * disjoint) / N + cfg.c3 * W;
    }

    bool save(const std::string& filepath) const {
        std::ofstream file(filepath);
        if (!file.is_open()) return false;

        file << "NODES " << nodes.size() << "\n";
        for (const auto& n : nodes) {
            file << n.id << " " << static_cast<int>(n.type) << " " << n.bias << "\n";
        }

        file << "CONNECTIONS " << connections.size() << "\n";
        file << std::fixed << std::setprecision(10);
        for (const auto& c : connections) {
            file << c.inNode << " " << c.outNode << " " 
                 << c.weight << " " << (c.enabled ? 1 : 0) << " " 
                 << c.innovation << "\n";
        }

        return true;
    }

    static Genome load(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
             throw std::runtime_error("No se pudo abrir el archivo de genoma NEAT.");
        }

        Genome g;
        std::string label;
        int count;

        // Leer NODES
        // Formato esperado: "NODES <count>"
        if (!(file >> label >> count) || label != "NODES") {
             // Intento de compatibilidad o error
             throw std::runtime_error("Formato inválido: Se esperaba 'NODES'");
        }

        g.nodes.reserve(count);
        for (int i = 0; i < count; ++i) {
            int id, typeInt;
            double bias;
            file >> id >> typeInt >> bias;
            NodeGene n;
            n.id = id;
            n.type = static_cast<NodeType>(typeInt);
            n.bias = bias;
            g.nodes.push_back(n);
        }

        // Leer CONNECTIONS
        // Formato esperado: "CONNECTIONS <count>"
        if (!(file >> label >> count) || label != "CONNECTIONS") {
            throw std::runtime_error("Formato inválido: Se esperaba 'CONNECTIONS'");
        }

        g.connections.reserve(count);
        for (int i = 0; i < count; ++i) {
            ConnectionGene c;
            int enabledInt;
            file >> c.inNode >> c.outNode >> c.weight >> enabledInt >> c.innovation;
            c.enabled = (enabledInt != 0);
            g.connections.push_back(c);
        }

        return g;
    }

private:
    bool hasConnection(int inId, int outId) const {
        for (const auto& c : connections) {
            if (c.inNode == inId && c.outNode == outId) {
                return true;
            }
        }
        return false;
    }

    bool createsCycle(int inId, int outId) const {
        // Check if there is a path from outId to inId (would create a cycle)
        std::unordered_map<int, std::vector<int>> adj;
        for (const auto& c : connections) {
            if (!c.enabled) continue;
            adj[c.inNode].push_back(c.outNode);
        }

        std::unordered_set<int> visited;
        std::vector<int> stack;
        stack.push_back(outId);

        while (!stack.empty()) {
            int node = stack.back();
            stack.pop_back();
            if (node == inId) return true;
            if (visited.insert(node).second) {
                for (int next : adj[node]) {
                    stack.push_back(next);
                }
            }
        }
        return false;
    }
};

} // namespace neat
} // namespace evonet

#endif
