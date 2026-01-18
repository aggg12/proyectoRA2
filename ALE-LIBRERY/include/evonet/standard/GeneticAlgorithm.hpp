/**
 * @file GeneticAlgorithm.hpp
 * @brief Motor evolutivo.
 */
#ifndef EVONET_GENETICALGORITHM_HPP
#define EVONET_GENETICALGORITHM_HPP

#include "Agent.hpp"
#include <vector>
#include <algorithm>
#include <random>

namespace evonet {

class GeneticAlgorithm {
private:
    std::vector<Agent> population;
    size_t generation;
    std::vector<size_t> topology;

    // Hiperparámetros
    double mutationRate;
    double crossoverRate;
    size_t elitismCount;
    
    // Función de activación configurada para toda la población
    NeuralNetwork::ActivationType activationTypeConfig;

    // Generador de números aleatorios
    std::mt19937 rng;

public:
    GeneticAlgorithm(size_t popSize,
                     const std::vector<size_t>& topo,
                     double mutRate = 0.05,
                     double crossRate = 0.5,
                     size_t elitism = 2)
        : generation(0),
          topology(topo),
          mutationRate(mutRate),
          crossoverRate(crossRate),
          elitismCount(elitism),
          rng(std::random_device{}()) {

        if (popSize == 0) {
            throw std::invalid_argument("El tamaño de población debe ser mayor que cero.");
        }
        
        // Por defecto Sigmoide
        activationTypeConfig = NeuralNetwork::ActivationType::SIGMOID;

        population.reserve(popSize);
        for (size_t i = 0; i < popSize; ++i) {
            population.emplace_back(topo);
            // Aseguramos que empiecen con la config correcta
            population.back().brain.setActivationType(activationTypeConfig);
        }
    }
    
    // Método para cambiar la función de activación de toda la especie
    void setActivationType(NeuralNetwork::ActivationType type) {
        activationTypeConfig = type;
        // Actualizar agentes existentes
        for(auto& agent : population) {
            agent.brain.setActivationType(activationTypeConfig);
        }
    }

    // Acceso a la población para que el usuario pueda evaluarla
    std::vector<Agent>& getPopulation() { return population; }
    const std::vector<Agent>& getPopulation() const { return population; }

    // Método principal: Avanzar una generación
    void evolve() {
        if (population.empty()) return;

        // 1. Ordenar población por fitness descendente
        std::sort(population.begin(), population.end(),
                  [](const Agent& a, const Agent& b) { return a.fitness > b.fitness; });

        std::vector<Agent> newPop;
        newPop.reserve(population.size());

        // 2. Elitismo: Preservar los mejores sin cambios
        for (size_t i = 0; i < elitismCount && i < population.size(); ++i) {
            newPop.push_back(population[i]);
        }

        // 3. Generar resto de la población
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        while (newPop.size() < population.size()) {
            // Selección
            const Agent& p1 = tournamentSelection();
            const Agent& p2 = tournamentSelection();

            Agent child(topology);
            // Asegurar que el hijo nazca con la configuración correcta
            child.brain.setActivationType(activationTypeConfig);

            // Cruce
            if (dist(rng) < crossoverRate) {
                child.brain.setGenome(crossover(p1.brain, p2.brain));
            } else {
                child.brain = p1.brain; // Copia directa si no hay cruce
            }

            // Mutación
            mutate(child.brain);

            // Reset fitness
            child.fitness = 0.0;
            newPop.push_back(child);
        }

        population = std::move(newPop);
        generation++;
    }

    size_t getGeneration() const { return generation; }

private:
    // Selección por Torneo: Escoge k individuos al azar y devuelve el mejor
    const Agent& tournamentSelection(size_t k = 5) {
        std::uniform_int_distribution<size_t> idxDist(0, population.size() - 1);
        const Agent* best = nullptr;

        for (size_t i = 0; i < k; ++i) {
            const Agent& candidate = population[idxDist(rng)];
            if (best == nullptr || candidate.fitness > best->fitness) {
                best = &candidate;
            }
        }
        return *best;
    }

    // Cruce Uniforme: Cada gen se elige al azar de uno de los padres
    std::vector<double> crossover(const NeuralNetwork& p1, const NeuralNetwork& p2) {
        std::vector<double> g1 = p1.getGenome();
        std::vector<double> g2 = p2.getGenome();
        if (g1.size() != g2.size()) {
            throw std::invalid_argument("Los genomas parentales tienen tamaños distintos.");
        }

        std::vector<double> childG;
        childG.reserve(g1.size());

        std::uniform_int_distribution<int> coin(0, 1);
        for (size_t i = 0; i < g1.size(); ++i) {
            childG.push_back(coin(rng) ? g1[i] : g2[i]);
        }
        return childG;
    }

    // Mutación Gaussiana: Pequeños ajustes a los pesos
    void mutate(NeuralNetwork& nn) {
        std::vector<double> genome = nn.getGenome();
        std::normal_distribution<double> noise(0.0, 0.5); // Media 0, DevStd 0.5
        std::uniform_real_distribution<double> chance(0.0, 1.0);

        for (auto& gene : genome) {
            if (chance(rng) < mutationRate) {
                gene += noise(rng);

                // Clamping opcional para evitar explosión
                if (gene > 10.0) gene = 10.0;
                if (gene < -10.0) gene = -10.0;
            }
        }
        nn.setGenome(genome);
    }
};

} // namespace evonet

#endif
