#include <iostream>
#include <vector>
#include "../../include/evonet/neat/Population.hpp"
#include "../../include/evonet/neat/Network.hpp"

int main() {
    using evonet::neat::NeatConfig;
    using evonet::neat::Population;
    using evonet::neat::Network;

    struct Sample {
        std::vector<double> input;
        std::vector<double> expected;
    };

    std::vector<Sample> data = {
        {{0, 0}, {0}},
        {{0, 1}, {1}},
        {{1, 0}, {1}},
        {{1, 1}, {0}}
    };

    NeatConfig cfg;
    cfg.inputCount = 2;
    cfg.outputCount = 1;
    // El resto usa los valores por defecto de NeatConfig.hpp
    
    Population pop(cfg);

    const int logEvery = 20;
    for (int gen = 0; gen < 1000; ++gen) {
        double bestFitness = -1e9;
        size_t bestIndex = 0;
        double sumFitness = 0.0;
        for (auto& genome : pop.getGenomes()) {
            Network net(genome);
            double errorSum = 0.0;

            for (const auto& s : data) {
                auto out = net.activate(s.input);
                double error = s.expected[0] - out[0];
                errorSum += error * error;
            }
            genome.fitness = 4.0 - errorSum;
            sumFitness += genome.fitness;
            if (genome.fitness > bestFitness) {
                bestFitness = genome.fitness;
                bestIndex = static_cast<size_t>(&genome - &pop.getGenomes()[0]);
            }
        }

        double avgFitness = sumFitness / static_cast<double>(pop.getGenomes().size());
        if (gen % logEvery == 0) {
            std::cout << "Gen " << gen
                      << " | Mejor: " << bestFitness
                      << " | Promedio: " << avgFitness
                      << std::endl;
        }

        if (bestFitness > 3.99) {
            std::cout << "\n¡Solución encontrada en generación " << gen << "!\n";
            std::cout << "Guardando modelo a 'neat_xor_model.txt'...\n";
            pop.getGenomes()[bestIndex].save("neat_xor_model.txt");
            break;
        }

        pop.evolve();
    }

    return 0;
}
