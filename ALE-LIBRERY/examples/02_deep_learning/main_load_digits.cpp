#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include "../../include/evonet/DigitRecognition.hpp"
#include "../../include/evonet/standard/NeuralNetwork.hpp"

int main(int argc, char** argv) {
    using evonet::DigitRecognition;
    using evonet::NeuralNetwork;

    // 1. Determinar el archivo del modelo (por defecto digits_model.txt)
    std::string modelPath = "digits_model.txt";
    if (argc > 1) {
        modelPath = argv[1];
    }

    std::cout << "=== Testeando Modelo Standard NN Guardado ===" << std::endl;
    std::cout << "Cargando modelo desde: " << modelPath << std::endl;

    // 2. Cargar la red neuronal
    // Inicializamos con topología dummy para evitar errores antes de cargar, 
    // aunque load sobreescribira esto.
    NeuralNetwork net({1,1}); 
    
    try {
        net = NeuralNetwork::load(modelPath);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] No se pudo cargar el modelo: " << e.what() << std::endl;
        std::cerr << "Asegurate de haber ejecutado el entrenamiento primero: make run_digits" << std::endl;
        return 1;
    }
    
    // Info de topologia
    const auto& topo = net.getTopology();
    std::cout << "Modelo cargado exitosamente." << std::endl;
    std::cout << "Topología: ";
    for(auto t : topo) std::cout << t << " ";
    std::cout << std::endl;

    // 3. Preparar los datos de prueba
    DigitRecognition task;
    const auto& data = task.getData();

    // 4. Evaluar
    int hits = 0;
    std::cout << "\nPredicciones del Modelo:" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    std::cout << " Dígito |   Predicción   | Confianza | Resultado" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    for (const auto& sample : data) {
        auto outputs = net.feedForward(sample.pixels);
        
        // Interpretar la salida (Argmax)
        int predicted = 0;
        double maxVal = -1.0;
        for(size_t i=0; i<outputs.size(); ++i) {
            if(outputs[i] > maxVal) {
                maxVal = outputs[i];
                predicted = (int)i;
            }
        }

        // Verificar vs dataset
        bool correct = (predicted == sample.label);
        if (correct) hits++;
        
        std::cout << "   " << sample.label << "    |       " << predicted << "        |" 
                  << std::fixed << std::setprecision(2) << std::setw(6) << maxVal 
                  << "   |  " << (correct ? "OK" : "FAIL") << std::endl;
    }
    std::cout << std::string(50, '-') << std::endl;

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
