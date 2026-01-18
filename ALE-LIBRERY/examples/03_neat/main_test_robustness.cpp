#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <random>
#include <chrono>
#include "../../include/evonet/DigitRecognition.hpp"
#include "../../include/evonet/neat/Genome.hpp"
#include "../../include/evonet/neat/Network.hpp"

// Función auxiliar para añadir ruido a un patrón
// rate: Probabilidad de invertir un bit (0.0 a 1.0)
std::vector<double> addNoise(const std::vector<double>& input, double rate, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<double> noisy = input;
    
    for (size_t i = 0; i < noisy.size(); ++i) {
        if (dist(rng) < rate) {
            // Invertir pixel: Si era 1.0 -> 0.0, Si era 0.0 -> 1.0
            noisy[i] = (noisy[i] > 0.5) ? 0.0 : 1.0;
        }
    }
    return noisy;
}

// Función para imprimir un digito (3x5) en consola
void printDigit(const std::vector<double>& pixels) {
    for (int r = 0; r < 5; ++r) {
        std::cout << "   ";
        for (int c = 0; c < 3; ++c) {
            double val = pixels[r * 3 + c];
            std::cout << (val > 0.5 ? "█" : "·");
        }
        std::cout << "\n";
    }
}

int main(int argc, char** argv) {
    using evonet::DigitRecognition;
    using evonet::neat::Genome;
    using evonet::neat::Network;

    std::string modelPath = "neat_digits_model.txt";
    if (argc > 1) {
        modelPath = argv[1];
    }

    // Inicializar generador random
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937 rng(seed);

    std::cout << "=== TEST DE ROBUSTEZ (Ruido) ===" << std::endl;
    std::cout << "Cargando modelo NEAT: " << modelPath << std::endl;

    // 1. Cargar modelo
    Genome genome;
    try {
        genome = Genome::load(modelPath);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }
    Network net(genome);

    // 2. Datos
    DigitRecognition task;
    const auto& data = task.getData();

    // 3. Configuración del test
    // Probaremos con diferentes niveles de ruido
    std::vector<double> noiseLevels = {0.05, 0.10, 0.20}; // 5%, 10%, 20% de pixeles corruptos

    for (double noiseRate : noiseLevels) {
        std::cout << "\n>>> NIVEL DE RUIDO: " << (int)(noiseRate * 100) << "% <<<" << std::endl;
        std::cout << "(Probabilidad de que un pixel se invierta)\n" << std::endl;

        int totalHits = 0;
        int trialsPerDigit = 10; // Probamos 10 variaciones por cada digito

        for (const auto& sample : data) {
            int digitHits = 0;
            
            // Solo imprimimos el primer ejemplo visualmente para no saturar consola
            bool visualShown = false;

            for (int t = 0; t < trialsPerDigit; ++t) {
                // Generar versión ruidosa
                std::vector<double> noisyInput = addNoise(sample.pixels, noiseRate, rng);

                // Predecir
                auto outputs = net.activate(noisyInput);
                
                int predicted = 0;
                double maxVal = -1.0;
                for(size_t i=0; i<outputs.size(); ++i) {
                    if(outputs[i] > maxVal) { maxVal = outputs[i]; predicted = (int)i; }
                }

                if (predicted == sample.label) digitHits++;

                // Visualización de muestra
                if (!visualShown && t == 0) {
                    std::cout << "Digito " << sample.label << " con ruido:\n";
                    printDigit(noisyInput);
                    std::cout << "Predicción: " << predicted << (predicted == sample.label ? " (OK)" : " (MAL)") << "\n";
                    visualShown = true;
                }
            }
            totalHits += digitHits;
            std::cout << "   Aciertos para el " << sample.label << ": " << digitHits << "/" << trialsPerDigit << "\n";
        }

        double totalAccuracy = (double)totalHits / (data.size() * trialsPerDigit) * 100.0;
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "Precision Total con " << (int)(noiseRate*100) << "% de ruido: " 
                  << std::fixed << std::setprecision(1) << totalAccuracy << "%" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
    }

    return 0;
}
