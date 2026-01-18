#ifndef EVONET_NEAT_POPULATION_HPP
#define EVONET_NEAT_POPULATION_HPP

#include "Genome.hpp"
#include "NeatConfig.hpp"
#include "Network.hpp"

#include <numeric>

namespace evonet {
namespace neat {

struct Species {
    int id = 0;
    std::vector<size_t> members;
    Genome representative;
    double bestFitness = 0.0;
    int staleness = 0;
};

class Population {
public:
    explicit Population(const NeatConfig& config)
        : cfg(config), rng(std::random_device{}()) {
        initializePopulation();
    }

    std::vector<Genome>& getGenomes() { return genomes; }
    const std::vector<Genome>& getGenomes() const { return genomes; }

    size_t getGeneration() const { return generation; }

    void speciate() {
        for (auto& s : species) {
            s.members.clear();
        }

        for (size_t i = 0; i < genomes.size(); ++i) {
            bool found = false;
            for (auto& s : species) {
                if (genomes[i].compatibilityDistance(s.representative, cfg) < cfg.compatibilityThreshold) {
                    s.members.push_back(i);
                    found = true;
                    break;
                }
            }
            if (!found) {
                Species s;
                s.id = static_cast<int>(species.size());
                s.representative = genomes[i];
                s.members.push_back(i);
                species.push_back(s);
            }
        }

        for (auto& s : species) {
            if (!s.members.empty()) {
                s.representative = genomes[s.members[0]];
            }
        }
    }

    void adjustFitness() {
        for (auto& s : species) {
            for (size_t idx : s.members) {
                genomes[idx].adjustedFitness = genomes[idx].fitness / static_cast<double>(s.members.size());
            }
        }
    }

    void evolve() {
        speciate();
        
        // 1. Dynamic Threshold Adjustment
        if (species.size() < cfg.targetSpeciesCount) {
             cfg.compatibilityThreshold = std::max(0.1, cfg.compatibilityThreshold - cfg.thresholdStep);
        } else if (species.size() > cfg.targetSpeciesCount) {
             cfg.compatibilityThreshold += cfg.thresholdStep;
        }

        // 2. Handle Stagnation (Aggressive)
        for (auto it = species.begin(); it != species.end(); ) {
            double currentMax = -1e9;
            for (size_t idx : it->members) {
                if (genomes[idx].fitness > currentMax) {
                    currentMax = genomes[idx].fitness;
                }
            }
            
            if (currentMax > it->bestFitness) {
                it->bestFitness = currentMax;
                it->staleness = 0;
            } else {
                it->staleness++;
            }
            
            // Si está estancada Y no es la única especie viva, mátala.
            if (it->staleness > cfg.dropOffAge && species.size() > 1) {
                it->members.clear();
                it->bestFitness = -1e9; 
            }
            
            if (it->members.empty()) {
                it = species.erase(it);
            } else {
                ++it;
            }
        }

        adjustFitness();

        std::vector<Genome> nextGen;
        nextGen.reserve(genomes.size());

        double totalAdjusted = 0.0;
        for (const auto& g : genomes) totalAdjusted += g.adjustedFitness;
        if (totalAdjusted == 0.0) totalAdjusted = 1.0;

        for (auto& s : species) {
            if (s.members.empty()) continue;

            std::vector<Genome*> members;
            members.reserve(s.members.size());
            for (size_t idx : s.members) members.push_back(&genomes[idx]);

            std::sort(members.begin(), members.end(),
                      [](const Genome* a, const Genome* b) { return a->fitness > b->fitness; });

            size_t survivors = std::max<size_t>(1, static_cast<size_t>(members.size() * cfg.survivalRate));

            size_t offspringCount = static_cast<size_t>(
                (sumAdjustedFitness(s) / totalAdjusted) * genomes.size());
            if (offspringCount == 0) offspringCount = 1;

            size_t eliteCount = std::min(cfg.elitism, survivors);
            for (size_t i = 0; i < eliteCount && nextGen.size() < genomes.size(); ++i) {
                nextGen.push_back(*members[i]);
            }

            std::uniform_real_distribution<double> chance(0.0, 1.0);
            while (nextGen.size() < genomes.size() && offspringCount-- > eliteCount) {
                Genome parent1 = *tournamentPick(members, survivors);
                Genome child;

                if (chance(rng) < cfg.crossoverRate && survivors > 1) {
                    Genome parent2 = *tournamentPick(members, survivors);
                    if (parent2.fitness > parent1.fitness) {
                        child = Genome::crossover(parent2, parent1, rng);
                    } else {
                        child = Genome::crossover(parent1, parent2, rng);
                    }
                } else {
                    child = parent1;
                }

                child.mutateWeights(cfg, rng);
                if (chance(rng) < cfg.addConnectionRate) {
                    child.mutateAddConnection(cfg, innovDb, rng);
                }
                if (chance(rng) < cfg.addNodeRate) {
                    child.mutateAddNode(innovDb, rng);
                }

                if (nextGen.size() < genomes.size()) {
                    nextGen.push_back(child);
                }
            }
        }

        while (nextGen.size() < genomes.size()) {
            nextGen.push_back(genomes[randomIndex(genomes.size())]);
        }

        genomes = std::move(nextGen);
        generation++;
    }

private:
    NeatConfig cfg;
    std::vector<Genome> genomes;
    std::vector<Species> species;
    InnovationDatabase innovDb;
    std::mt19937 rng;
    size_t generation = 0;

    std::vector<int> inputIds;
    int biasId = -1;
    std::vector<int> outputIds;

    void initializePopulation() {
        inputIds.clear();
        outputIds.clear();

        for (size_t i = 0; i < cfg.inputCount; ++i) {
            inputIds.push_back(innovDb.createNodeId());
        }
        biasId = innovDb.createNodeId();
        for (size_t i = 0; i < cfg.outputCount; ++i) {
            outputIds.push_back(innovDb.createNodeId());
        }

        genomes.reserve(cfg.populationSize);
        for (size_t i = 0; i < cfg.populationSize; ++i) {
            genomes.emplace_back(inputIds, biasId, outputIds, innovDb, rng);
        }
    }

    double sumAdjustedFitness(const Species& s) const {
        double sum = 0.0;
        for (size_t idx : s.members) {
            sum += genomes[idx].adjustedFitness;
        }
        return sum;
    }

    Genome* tournamentPick(const std::vector<Genome*>& members, size_t count) {
        std::uniform_int_distribution<size_t> pick(0, count - 1);
        Genome* best = nullptr;
        for (size_t i = 0; i < 3; ++i) {
            Genome* candidate = members[pick(rng)];
            if (!best || candidate->fitness > best->fitness) best = candidate;
        }
        return best;
    }

    size_t randomIndex(size_t max) {
        std::uniform_int_distribution<size_t> dist(0, max - 1);
        return dist(rng);
    }
};

} // namespace neat
} // namespace evonet

#endif
