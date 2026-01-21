#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <random>
#include <chrono>
#include "../../include/evonet/DigitRecognition.hpp"
#include "../../include/evonet/standard/NeuralNetwork.hpp"

// Función auxiliar para añadir ruido a un patrón
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

// Función para imprimir un digito (5x7) en consola
void printDigit(const std::vector<double>& pixels) {
    if (pixels.size() != 35) {
        std::cout << " [Error: Tamaño de imagen incorrecto (" << pixels.size() << "), se esperaba 35]\n";
        return;
    }
    for (int r = 0; r < 7; ++r) {
        std::cout << "   ";
        for (int c = 0; c < 5; ++c) {
            double val = pixels[r * 5 + c];
            std::cout << (val > 0.5 ? "█" : "·");
        }
        std::cout << "\n";
    }
}

int main(int argc, char** argv) {
    using evonet::DigitRecognition;
    using evonet::NeuralNetwork;

    std::string modelPath = "digits_model.txt";
    if (argc > 1) {
        modelPath = argv[1];
    }

    // Inicializar generador random
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937 rng(seed);

    std::cout << "=== TEST DE ROBUSTEZ STANDARD (Ruido) ===" << std::endl;
    std::cout << "Cargando modelo Standard NN: " << modelPath << std::endl;

    // 1. Cargar modelo 
    NeuralNetwork net({1,1}); // Objeto dummy
    try {
        net = NeuralNetwork::load(modelPath);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        std::cerr << "Ejecuta primero: make run_digits" << std::endl;
        return 1;
    }

    // 2. Datos
    DigitRecognition task;
    const auto& data = task.getData();

    // 3. Configuración del test
    // ESCALA DE DIFICULTAD (Fácil a Medio)
    // 0.00: Test de Sanidad (¿Recuerda lo básico?)
    // 0.20: Límite actual de entrenamiento
    std::vector<double> noiseLevels = {0.00, 0.05, 0.10, 0.15, 0.20}; 

    for (double noiseRate : noiseLevels) {
        std::cout << "\n>>> NIVEL DE RUIDO: " << (int)(noiseRate * 100) << "% <<<" << std::endl;
        if (noiseRate == 0.0) std::cout << "(Datos Perfectos - Control de Calidad)\n" << std::endl;
        else std::cout << "(Probabilidad de pixel invertido)\n" << std::endl;

        int totalHits = 0;
        int trialsPerDigit = 100; // Muestras por digito para estadística fiable 

        for (const auto& sample : data) {
            int digitHits = 0;
            bool visualShown = false;

            for (int t = 0; t < trialsPerDigit; ++t) {
                // Generar versión ruidosa
                std::vector<double> noisyInput = addNoise(sample.pixels, noiseRate, rng);

                // Predecir (Usamos feedForward en Standard NN)
                auto outputs = net.feedForward(noisyInput);
                
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
