#ifndef EVONET_NEAT_CONFIG_HPP
#define EVONET_NEAT_CONFIG_HPP

#include <cstddef>

namespace evonet {
namespace neat {

struct NeatConfig {
    size_t populationSize = 150;
    size_t inputCount = 2;
    size_t outputCount = 1;

    double weightMutationRate = 0.8;
    double weightPerturbRate = 0.9;
    double weightPerturbStd = 0.5;
    double weightReplaceRange = 1.0;

    double addConnectionRate = 0.05;
    double addNodeRate = 0.03;
    double crossoverRate = 0.75;

    size_t elitism = 2;
    double survivalRate = 0.5;

    // Compatibility distance coefficients
    double c1 = 1.0;
    double c2 = 1.0;
    double c3 = 0.4;
    double compatibilityThreshold = 3.0;

    // Dynamic Thresholding & Stagnation (Standard)
    size_t targetSpeciesCount = 20;    // Standard diversity
    double thresholdStep = 0.05;       // Gentle adjustment
    int dropOffAge = 20;               // Give species time to optimize

    // Max tries for adding a new connection
    size_t addConnectionTries = 30;
};

} // namespace neat
} // namespace evonet

#endif
