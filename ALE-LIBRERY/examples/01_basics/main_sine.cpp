#include <iostream>
#include <vector>
#include <cmath>
#include "../../include/evonet/standard/GeneticAlgorithm.hpp"

int main() {
    using evonet::GeneticAlgorithm;

    struct TrainingData {
        std::vector<double> input;
        std::vector<double> expected;
    };

    std::vector<TrainingData> sineData;
    for (double x = 0.0; x <= 3.14159; x += 0.1) {
        sineData.push_back({{x}, {std::sin(x)}});
    }

    // Usando valores por defecto: Mutacion=0.05, Cruce=0.5
    GeneticAlgorithm ga(150, {1, 8, 1});

    std::cout << "Iniciando evolución para aproximar sin(x) (Params por defecto)..." << std::endl;

    const int logEvery = 20;
    for (int gen = 0; gen < 1000; ++gen) {
        auto& population = ga.getPopulation();

        double bestFitness = -1e9;
        size_t bestIndex = 0;
        double sumFitness = 0.0;
        for (auto& agent : population) {
            double mse = 0.0;
            for (const auto& d : sineData) {
                auto output = agent.brain.feedForward(d.input);
                double error = d.expected[0] - output[0];
                mse += error * error;
            }
            mse /= static_cast<double>(sineData.size());
            agent.fitness = 1.0 / (1.0 + mse);
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

        if (bestFitness > 0.99) {
            std::cout << "\nAproximación aceptable encontrada en generación " << gen << ".\n";
            std::cout << "Guardando modelo a 'sine_model.txt'...\n";
            population[bestIndex].brain.save("sine_model.txt");
            break;
        }

        ga.evolve();
    }

    return 0;
}
