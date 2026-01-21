#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#include "../../include/evonet/standard/GeneticAlgorithm.hpp"
#include "../../include/evonet/DigitRecognition.hpp"

// Función auxiliar para añadir ruido (Data Augmentation en entrenamiento)
std::vector<double> addTrainingNoise(const std::vector<double>& input, double rate, std::mt19937& rng) {
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
    using evonet::GeneticAlgorithm;
    using evonet::DigitRecognition;

    // Semilla aleatoria
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937 rng(seed);

    // DIGIT RECOGNITION (5x7 Pixels -> 10 Classes)
    // 35 Inputs -> [Hidden Layer 1] -> [Hidden Layer 2] -> 10 Outputs
    // DEEP LEARNING: Añadimos una segunda capa oculta para abstracción jerárquica.
    // Capa 1 (60): Detecta rasgos (bordes, curvas).
    // Capa 2 (30): Combina rasgos (círculos, cruces).
    GeneticAlgorithm ga(200, {35, 60, 30, 10}); 
    
    // GUARDAR EL CAMPEÓN HISTÓRICO
    // (Para no perder el mejor modelo si la mutación lo estropea al final)
    evonet::Agent globalBest = ga.getPopulation()[0];
    globalBest.fitness = -1e9;
    double globalBestNoiseLevel = -1.0; // Récord de dificultad superada

    DigitRecognition task;
    const auto& data = task.getData();

    std::cout << "Iniciando evolución para RECONOCIMIENTO DE DÍGITOS (Standard) - MODO ESPARTANO..." << std::endl;

    const int logEvery = 20;
    
    // Nivel de ruido durante entrenamiento
    double noiseLevel = 0.01; 
    
    // Contador de victorias consecutivas
    int consecutiveWins = 0;

    int gen = 0;
    std::cout << "Presiona Ctrl+C para detener el entrenamiento. Se guarda automaticamente el mejor modelo.\n";
    while (true) {
        gen++;
        auto& population = ga.getPopulation();

        // CURRICULUM ADAPTATIVO:
        // Solo subimos la dificultad si dominan el nivel actual (Aciertos > 9 de manera consistente)
        // Esto evita "ahogar" a las redes con ruido imposible antes de tiempo.
        // Pero mantenemos un aumento muy lento garantizado para evitar estancamiento total.
        
        // (Nota: La subida se decide tras evaluar la poblacion, ver abajo)

        double bestFitness = -1e9;
        int maxHits = 0;
        
        double sumFitness = 0.0;
        
        for (size_t k = 0; k < population.size(); ++k) {
            auto& agent = population[k];
            double totalError = 0.0;
            int hits = 0;

            // En cada generación, generamos una versión LIGERAMENTE distinta del dataset
            // Esto evita que memoricen una sola foto y los obliga a generalizar.
            for (const auto& sample : data) {
                
                // --- PROTECCION CONTRA OLVIDO CATASTROFICO ---
                // En vez de aplicar siempre 'noiseLevel', elegimos un ruido al azar entre 0 y el Nivel Actual.
                // Así la red ve una mezcla constante de ejemplos LIMPIOS, FACILES y DIFICILES.
                double currentSampleNoise = 0.0;
                if (noiseLevel > 0.0001) {
                     std::uniform_real_distribution<double> noiseDist(0.0, noiseLevel);
                     currentSampleNoise = noiseDist(rng);
                }

                // Aplicamos el ruido calculado para esta muestra especifica
                std::vector<double> inputs = addTrainingNoise(sample.pixels, currentSampleNoise, rng);

                auto outputs = agent.brain.feedForward(inputs);
                
                // 1. Calcular error cuadrático y predicción
                double sampleError = 0.0;
                int predictedLabel = -1;
                double maxVal = -1e9;

                for (size_t i = 0; i < 10; ++i) {
                    double diff = sample.targets[i] - outputs[i];
                    sampleError += diff * diff;

                    if (outputs[i] > maxVal) {
                        maxVal = outputs[i];
                        predictedLabel = static_cast<int>(i);
                    }
                }
                totalError += sampleError;

                // 2. Contar aciertos
                if (predictedLabel == sample.label) {
                    hits++;
                }
            }

            // Fitness: Aciertos primero, luego error
            agent.fitness = (hits * 10.0) + (10.0 - totalError);
            
            sumFitness += agent.fitness;
            
            if (agent.fitness > bestFitness) {
                bestFitness = agent.fitness;
                maxHits = hits;
            }
        }

        // --- LÓGICA DE CURRICULUM ADAPTATIVO (CONSECUTIVO) ---
        // Si dominan el ruido actual (>= 9 aciertos) durante VARIAS generaciones seguidas, subimos nivel.
        if (maxHits >= 9 && noiseLevel < 0.20) {
            consecutiveWins++;
            // Requerimos 10 confirmaciones seguidas para asegurar que no es suerte
            if (consecutiveWins >= 10) {
                 noiseLevel += 0.02; // Subida firme
                 std::cout << " [!!!] Dominio consistente (" << consecutiveWins << " gens). Subiendo dificultad a " << int(noiseLevel*100) << "%" << std::endl;
                 consecutiveWins = 0; // Resetear contador para el nuevo nivel
            }
        } else {
            // Si fallan una vez, reseteamos el contador (o lo bajamos)
            consecutiveWins = 0;

        }
        // ---------------------------------------

        // Actualizar Campeón Histórico (Lógica Inteligente)
        // Valoramos mas sobrevivir al ruido alto que un fitness perfecto en ruido bajo
        bool isNewRecord = false;
        
        // Si estamos en un nivel de ruido superior al record anterior...
        if (noiseLevel > globalBestNoiseLevel) {
            // ...y el desempeño es aceptable (no guardamos basura solo por ser dificil)
            if (maxHits >= 8) {
                isNewRecord = true;
            }
        } 
        // Si estamos en el mismo nivel, desempata el fitness
        else if (noiseLevel == globalBestNoiseLevel) {
            if (bestFitness > globalBest.fitness) {
                isNewRecord = true;
            }
        }

        if (isNewRecord) {
            for (const auto& ag : population) {
                if (ag.fitness == bestFitness) {
                    globalBest = ag; 
                    globalBestNoiseLevel = noiseLevel;
                    std::cout << " [Nuevo CAMPEON] Ruido: " << int(noiseLevel*100) << "% | Fitness: " << bestFitness << std::endl;
                    
                    // Guardar inmediatamente el mejor modelo encontrado
                    globalBest.brain.save("digits_model.txt");
                    std::cout << "   (Modelo guardado en 'digits_model.txt')" << std::endl;
                    
                    break; 
                }
            }
        }

        if (gen % logEvery == 0) {
            std::cout << "Gen " << std::setw(4) << gen
                      << " | Fitness: " << std::fixed << std::setprecision(2) << bestFitness
                      << " | Aciertos: " << maxHits << "/10"
                      << std::endl;
        }

        if (maxHits == 10 && bestFitness > 109.0) {
            // Ya es perfecto con el nivel de ruido actual
            // NO SALIMOS, SEGUIMOS PARA ROBUSTEZ TOTAL
            // Solo notificamos
            if (gen % 500 == 0) {
                std::cout << " [Perfecto] (Continuando entrenamiento para robustez...)" << std::endl;
            }
        }

        ga.evolve();
    }
    return 0;
}
    // El bucle es infinito, el return de abajo nunca se alcanza en modo infinito
    // pero lo dejamos por corrección sintáctica y si cambiamos la condición.

    /*
    std::cout << "Finalizado (Limite alcanzado). Guardando EL MEJOR modelo historico a 'digits_model.txt'...\n";
    std::cout << "Best Fitness Historico: " << globalBest.fitness << std::endl;
    
    globalBest.brain.save("digits_model.txt");
    */
