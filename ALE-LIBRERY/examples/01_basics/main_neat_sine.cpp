#include <iostream>
#include <vector>
#include <cmath>
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

    std::vector<Sample> data;
    for (double x = 0.0; x <= 3.14159; x += 0.1) {
        data.push_back({{x}, {std::sin(x)}});
    }

    NeatConfig cfg;
    cfg.inputCount = 1;
    cfg.outputCount = 1;
    // El resto usa los valores por defecto de NeatConfig.hpp

    Population pop(cfg);

    const int logEvery = 20;
    for (int gen = 0; gen < 1000; ++gen) {
        double bestFitness = -1e9;
        size_t bestIndex = 0;
        double sumFitness = 0.0;

        auto& genomes = pop.getGenomes();
        for (size_t i = 0; i < genomes.size(); ++i) {
            auto& genome = genomes[i];
            Network net(genome);
            double mse = 0.0;
            for (const auto& s : data) {
                auto out = net.activate(s.input);
                double error = s.expected[0] - out[0];
                mse += error * error;
            }
            mse /= static_cast<double>(data.size());
            genome.fitness = 1.0 / (1.0 + mse);
            sumFitness += genome.fitness;
            if (genome.fitness > bestFitness) {
                bestFitness = genome.fitness;
                bestIndex = i;
            }
        }

        double avgFitness = sumFitness / static_cast<double>(pop.getGenomes().size());
        if (gen % logEvery == 0) {
            std::cout << "Gen " << gen
                      << " | Mejor: " << bestFitness
                      << " | Promedio: " << avgFitness
                      << std::endl;
        }

        if (bestFitness > 0.99) {
            std::cout << "\nAproximación aceptable encontrada en generación " << gen << ".\n";
            std::cout << "Guardando modelo a 'neat_sine_model.txt'...\n";
            genomes[bestIndex].save("neat_sine_model.txt");
            break;
        }

        pop.evolve();
    }

    return 0;
}
