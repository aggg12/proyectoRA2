/**
 * @file Agent.hpp
 * @brief Estructura base para agentes evolutivos.
 */
#ifndef EVONET_AGENT_HPP
#define EVONET_AGENT_HPP

#include "NeuralNetwork.hpp"

namespace evonet {

struct Agent {
    NeuralNetwork brain;
    double fitness;

    explicit Agent(const std::vector<size_t>& topo)
        : brain(topo), fitness(0.0) {}

    bool operator>(const Agent& other) const {
        return fitness > other.fitness;
    }
};

} // namespace evonet

#endif
