#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include "../../include/evonet/DigitRecognition.hpp"
#include "../../include/evonet/neat/Genome.hpp"
#include "../../include/evonet/neat/Network.hpp"

int main(int argc, char** argv) {
    using evonet::DigitRecognition;
    using evonet::neat::Genome;
    using evonet::neat::Network;

    // 1. Determinar el archivo del modelo (por defecto neat_digits_model.txt)
    std::string modelPath = "neat_digits_model.txt";
    if (argc > 1) {
        modelPath = argv[1];
    }

    std::cout << "=== Testeando Modelo NEAT Guardado ===" << std::endl;
    std::cout << "Cargando modelo desde: " << modelPath << std::endl;

    // 2. Cargar el genoma
    Genome genome;
    try {
        genome = Genome::load(modelPath);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] No se pudo cargar el modelo: " << e.what() << std::endl;
        std::cerr << "Asegurate de haber ejecutado el entrenamiento primero: make run_neat_digits" << std::endl;
        return 1;
    }
    std::cout << "Modelo cargado exitosamente." << std::endl;
    std::cout << "Nodos: " << genome.nodes.size() << ", Conexiones: " << genome.connections.size() << std::endl;

    // 3. Crear la red neuronal (Genotipo -> Fenotipo)
    Network net(genome);

    // 4. Preparar los datos de prueba
    DigitRecognition task;
    const auto& data = task.getData();

    // 5. Evaluar
    int hits = 0;
    std::cout << "\nPredicciones del Modelo:" << std::endl;
    std::cout << std::string(65, '-') << std::endl;
    std::cout << " Dígito | Predicción | Confianza | 2do Mejor (Conf) | Resultado" << std::endl;
    std::cout << std::string(65, '-') << std::endl;

    for (const auto& sample : data) {
        auto outputs = net.activate(sample.pixels);
        
        // Interpretar la salida (Argmax y 2nd Best)
        int predicted = -1;
        double maxVal = -1e9;
        
        int runnerUp = -1;
        double runnerUpVal = -1e9;

        // Nota: NEAT puede tener menos outputs si no ha evolucionado todos, manejamos eso
        size_t limit = (outputs.size() < 10) ? outputs.size() : 10;

        for(size_t i=0; i<10; ++i) {
            double val = (i < limit) ? outputs[i] : 0.0;

            if(val > maxVal) {
                // El antiguo mejor pasa a ser segundo
                runnerUpVal = maxVal;
                runnerUp = predicted;
                
                maxVal = val;
                predicted = (int)i;
            } else if (val > runnerUpVal) {
                runnerUpVal = val;
                runnerUp = (int)i;
            }
        }

        // Verificar vs dataset (ya sabemos que label es el indice 1 en targets)
        bool correct = (predicted == sample.label);
        if (correct) hits++;
        
        std::cout << "   " << sample.label << "    |      " << predicted << "     |" 
                  << std::fixed << std::setprecision(2) << std::setw(6) << maxVal << "    |"
                  << "    " << runnerUp << " (" << std::setw(4) << runnerUpVal << ")   |  " 
                  << (correct ? "OK" : "FAIL") << std::endl;
    }
    std::cout << std::string(65, '-') << std::endl;

    double accuracy = (double)hits / data.size() * 100.0;
    std::cout << "\nResumen:" << std::endl;
    std::cout << "Aciertos Totales: " << hits << "/" << data.size() << std::endl;
    std::cout << "Precisión Final: " << accuracy << "%" << std::endl;

    if (accuracy == 100.0) {
        std::cout << "¡Excelente! El modelo ha aprendido todos los dígitos." << std::endl;
    } else {
        std::cout << "El modelo aún puede mejorar. Intenta entrenar más tiempo." << std::endl;
    }

    return 0;
}
