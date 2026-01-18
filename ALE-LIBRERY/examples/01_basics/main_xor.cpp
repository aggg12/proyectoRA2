#include <iostream>
#include <vector>
#include "../../include/evonet/standard/GeneticAlgorithm.hpp"

int main() {
    using evonet::GeneticAlgorithm;

    struct TrainingData {
        std::vector<double> input;
        std::vector<double> expected;
    };

    std::vector<TrainingData> data = {
        {{0, 0}, {0}},
        {{0, 1}, {1}},
        {{1, 0}, {1}},
        {{1, 1}, {0}}
    };

    // Usando valores por defecto: Mutacion=0.05, Cruce=0.5
    GeneticAlgorithm ga(200, {2, 4, 1});

    std::cout << "Iniciando evolución para XOR (Params por defecto)..." << std::endl;

    const int logEvery = 20;
    for (int gen = 0; gen < 1000; ++gen) {
        auto& population = ga.getPopulation();

        double bestFitness = -1e9;
        size_t bestIndex = 0;
        double sumFitness = 0.0;
        for (auto& agent : population) {
            double errorSum = 0.0;
            for (const auto& d : data) {
                auto output = agent.brain.feedForward(d.input);
                double error = d.expected[0] - output[0];
                errorSum += error * error;
            }
            agent.fitness = 4.0 - errorSum;
            sumFitness += agent.fitness;
            if (agent.fitness > bestFitness) {
                bestFitness = agent.fitness;
                bestIndex = static_cast<size_t>(&agent - &population[0]);
            }
        }

        double avgFitness = sumFitness / static_cast<double>(population.size());
        if (gen % logEvery == 0) {
            std::cout << "Gen " << gen
                      << " | Mejor: " << bestFitness
                      << " | Promedio: " << avgFitness
                      << std::endl;
        }

        if (bestFitness > 3.99) {
            std::cout << "\n¡Solución encontrada en generación " << gen << "!\n";
            std::cout << "Resultados del mejor agente:\n";
            const auto& best = population[bestIndex];
            for (const auto& d : data) {
                auto out = best.brain.feedForward(d.input);
                std::cout << "In: (" << d.input[0] << ", " << d.input[1]
                          << ") -> Out: " << out[0]
                          << " (Exp: " << d.expected[0] << ")\n";
            }
            std::cout << "Guardando modelo a 'xor_model.txt'...\n";
            best.brain.save("xor_model.txt");
            break;
        }

        ga.evolve();
    }

    return 0;
}
