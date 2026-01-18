#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include "../../include/evonet/DigitRecognition.hpp"
#include "../../include/evonet/neat/Population.hpp"
#include "../../include/evonet/neat/Network.hpp"

// Función auxiliar para ruido en NEAT (Training)
std::vector<double> addNeatNoise(const std::vector<double>& input, double rate, std::mt19937& rng) {
    if (rate <= 0.0) return input;
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<double> noisy = input;
    for (size_t i = 0; i < noisy.size(); ++i) {
        if (dist(rng) < rate) {
            noisy[i] = (noisy[i] > 0.5) ? 0.0 : 1.0;
        }
    }
    return noisy;
}

int main() {
    using evonet::DigitRecognition;
    using evonet::neat::NeatConfig;
    using evonet::neat::Population;
    using evonet::neat::Network;

    NeatConfig cfg;
    cfg.inputCount = 35; // 5x7 Grid
    cfg.outputCount = 10; // 0-9

    // CONFIGURACIÓN "GOLDILOCKS" (La mejor apuesta a largo plazo)
    // Equilibrio entre exploración agresiva y estabilidad para aprender
    
    cfg.populationSize = 300;            // Buen balance velocidad/diversidad
    // CONFIGURACIÓN "GOLDILOCKS" (La mejor apuesta a largo plazo)
    // Equilibrio entre exploración agresiva y estabilidad para aprender
    
    cfg.populationSize = 300;            // Buen balance velocidad/diversidad
    
    // Tasas Estructurales: Necesitamos crecer, pero sin romper la red cada turno
    cfg.addConnectionRate = 0.40;        // 40% de probabilidad de nuevas sinapsis
    cfg.addNodeRate = 0.10;              // 10% de probabilidad de nuevas neuronas (crucial para digits)
    
    // Ajuste de pesos (El aprendizaje real)
    cfg.weightMutationRate = 0.8;        // Muy frecuente
    cfg.weightPerturbStd = 0.5;          // Perturbación media (ni muy tímida 0.1, ni muy loca 1.0)

    cfg.compatibilityThreshold = 3.0;    
    cfg.survivalRate = 0.20;             // El top 20% pasa sus genes (Estándar robusto)

    Population pop(cfg);
    DigitRecognition task;
    const auto& data = task.getData();
    
    // Variables de Estado (Curriculum + Smart Save)
    double noiseLevel = 0.00;
    int consecutiveWins = 0;
    double maxNoiseReached = 0.00; // Track record
    double bestFitnessAtMaxNoise = -1e9; // Track fitness at record

    // Generador random para ruido
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937 rng(seed);
    
    // Nivel base de ruido para entrenamiento (Data Augmentation)
    // double noiseLevel = 0.02; // YA DEFINIDO ARRIBA
    
    // Contador de victorias consecutivas
    // int consecutiveWins = 0; // YA DEFINIDO ARRIBA
    
    const int logEvery = 20;

    std::cout << "Iniciando NEAT para DIGIT RECOGNITION (Optimizada + ROBUSTEZ)..." << std::endl;
    std::cout << "Presiona Ctrl+C para detener. El modelo se guarda automaticamente tras cada record.\n";

    // Bucle infinito: NEAT toma tiempo.
    int gen = 0;
    while(true) {
        gen++;
        
        // Curriculum gestionado despues de la evaluacion

        double bestFitness = -1e9;
        int maxHits = 0;

        auto& genomes = pop.getGenomes();
        for (size_t k = 0; k < genomes.size(); ++k) {
            auto& genome = genomes[k];
            Network net(genome);
            
            double totalError = 0.0;
            int hits = 0;

            for (const auto& sample : data) {
                 // Aplicamos RUIDO al input para que no memoricen
                 std::vector<double> inputs = addNeatNoise(sample.pixels, noiseLevel, rng);
                 auto outputs = net.activate(inputs);

                 double sampleError = 0.0;
                 int predictedLabel = -1;
                 double maxVal = -1e9;

                 // Aseguramos que outputs tenga tamano 10 (puede ser menor si la red esta rota o creciendo)
                 size_t limit = (outputs.size() < 10) ? outputs.size() : 10;

                 for (size_t i = 0; i < 10; ++i) {
                     double val = (i < limit) ? outputs[i] : 0.0;
                     
                     double diff = sample.targets[i] - val;
                     sampleError += diff * diff;

                     if (val > maxVal) {
                         maxVal = val;
                         predictedLabel = static_cast<int>(i);
                     }
                 }
                 totalError += sampleError;

                 if (predictedLabel == sample.label) {
                     hits++;
                 }
            }

            genome.fitness = (hits * 10.0) + (10.0 - totalError);
            if (genome.fitness > bestFitness) {
                bestFitness = genome.fitness;
                maxHits = hits;
            }
        }

        if (gen % logEvery == 0) {
             std::cout << "Gen " << std::setw(4) << gen
                      << " | Fitness: " << std::fixed << std::setprecision(2) << bestFitness
                      << " | Aciertos: " << maxHits << "/10"
                      << std::endl;
        }

        // --- CURRICULUM ADAPTATIVO (NEAT - CONSECUTIVO & PACIENTE) ---
        // Si dominan el ruido actual (>= 9 aciertos) varias veces seguidas, subimos nivel.
        // NEAT necesita tiemp para estabilizar nuevas conexiones, así que somos pacientes.
        if (maxHits >= 9 && noiseLevel < 0.20) {
             consecutiveWins++;
             // Pedimos mucha consistencia antes de subir, para no "quemar" especies inmaduras
             if (consecutiveWins >= 25) { 
                 noiseLevel += 0.01; // Subimos despacito, solo 1%
                 std::cout << " [!!!] Especie muy robusta (" << consecutiveWins << " gens). Subiendo dificultad a " << int(noiseLevel*100) << "%" << std::endl;
                 consecutiveWins = 0;
             }
        } else {
             // Si fallan, reseteamos el contador. NO SUBIMOS POR TIEMPO.
             // En NEAT, subir la presión artificialmente (por tiempo) suele extinguir las especies prometedoras.
             consecutiveWins = 0; 
        }

        // --- Guardado Inteligente (Smart Saving 2.0) ---
        // Igual que en Deep Learning: Guardar solo si mejora el Nivel de Ruido O el Fitness en ese nivel.
        bool isNewRecord = false;

        if (noiseLevel > maxNoiseReached) {
            maxNoiseReached = noiseLevel;
            bestFitnessAtMaxNoise = bestFitness;
            isNewRecord = true;
            std::cout << " [SAVE] Nuevo Récord de Ruido NEAT: " << int(maxNoiseReached*100) << "%!" << std::endl;
        } 
        else if (noiseLevel == maxNoiseReached) {
            if (bestFitness > bestFitnessAtMaxNoise) {
                bestFitnessAtMaxNoise = bestFitness;
                isNewRecord = true;
                // std::cout << " [SAVE] Mejor Fitness NEAT." << std::endl; // Opcional, para no spaamear
            }
        }

        if (isNewRecord) {
             // Guardar el mejor
             auto& genomes = pop.getGenomes();
             // Buscar el mismo que imprimimos arriba
             size_t bestIdx = 0;
             double bFit = -1e9;
             for(size_t i=0; i<genomes.size(); ++i) {
                 if(genomes[i].fitness > bFit) {
                     bFit = genomes[i].fitness;
                     bestIdx = i;
                 }
             }
             // Crear Network a partir del genoma para guardarlo
             // Usamos directamente el método save del genoma
             genomes[bestIdx].save("neat_digits_model.txt");
             // std::cout << " [SAVE] Modelo guardado." << std::endl;
        }
        // ------------------------------------

        if (maxHits == 10 && bestFitness > 109.0) {
            // No salimos. Seguimos entrenando con el ruido creciente
            if (gen % 500 == 0) {
                 std::cout << " [Perfecto] (Continuando entrenamiento experto...)" << std::endl;
            }
        }

        pop.evolve();
    }
    
    // std::cout << "Finalizado...\n";
    // ...
    // finalGenomes[finalBestIndex].save("neat_digits_model.txt");

    return 0;
}
