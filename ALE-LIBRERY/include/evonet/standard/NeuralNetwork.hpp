/**
 * @file NeuralNetwork.hpp
 * @brief Clase principal de la red neuronal Feed-Forward con persistencia.
 */
#ifndef EVONET_NEURALNETWORK_HPP
#define EVONET_NEURALNETWORK_HPP

#include "../Matrix.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <functional>
#include <stdexcept>
#include <iomanip>

namespace evonet {

// Funciones de activación comunes
namespace Activation {
    inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
    inline double relu(double x) { return x > 0 ? x : 0; }
    inline double tanh(double x) { return std::tanh(x); }
    inline double linear(double x) { return x; }
}

class NeuralNetwork {
public:
    enum class ActivationType {
        SIGMOID = 0,
        RELU = 1,
        TANH = 2,
        LINEAR = 3
    };

private:
    std::vector<size_t> topology;
    std::vector<Matrix> weights;
    std::vector<Matrix> biases;
    std::function<double(double)> activationFunc;
    ActivationType currentActivationType; // Para guardar el ID

public:
    // Constructor: Define topología y aleatoriza pesos iniciales
    explicit NeuralNetwork(const std::vector<size_t>& topo) : topology(topo) {
        if (topology.size() < 2) {
            throw std::invalid_argument("La topología debe tener al menos 2 capas.");
        }
        for (size_t n : topology) {
            if (n == 0) {
                throw std::invalid_argument("Las capas no pueden tener 0 neuronas.");
            }
        }

        setActivationType(ActivationType::SIGMOID); // Default

        for (size_t i = 0; i < topology.size() - 1; ++i) {
            // Pesos entre capa i e i+1
            // Dimensiones: filas = neuronas destino, cols = neuronas origen
            weights.emplace_back(topology[i + 1], topology[i], true);

            // Sesgos para capa i+1
            biases.emplace_back(topology[i + 1], 1, true);
        }
    }

    // Configurar función de activación por Tipo (Recomendado)
    void setActivationType(ActivationType type) {
        currentActivationType = type;
        switch(type) {
            case ActivationType::SIGMOID: activationFunc = Activation::sigmoid; break;
            case ActivationType::RELU:    activationFunc = Activation::relu; break;
            case ActivationType::TANH:    activationFunc = Activation::tanh; break;
            case ActivationType::LINEAR:  activationFunc = Activation::linear; break;
        }
    }

    // Configurar función de activación manual (Avanzado, no se guarda el ID)
    void setActivation(const std::function<double(double)>& func) {
        activationFunc = func;
        // No cambiamos el ID porque no sabemos cual es
    }

    // Getter del tipo actual
    ActivationType getActivationType() const { return currentActivationType; }

    // Propagación hacia adelante
    std::vector<double> feedForward(const std::vector<double>& inputVals) const {
        if (inputVals.size() != topology.front()) {
            throw std::invalid_argument("Tamaño de entrada no coincide con la capa de entrada.");
        }

        // Convertir vector de entrada a Matriz columna
        Matrix inputs(topology.front(), 1);
        size_t offset = 0;
        inputs.fromVector(inputVals, offset);

        Matrix curr = inputs;

        for (size_t i = 0; i < weights.size(); ++i) {
            // Operación lineal: Z = W * A + B
            Matrix z = weights[i].multiply(curr);
            z.add(biases[i]);

            // Operación no lineal: A = Activation(Z)
            curr = z.map(activationFunc);
        }

        return curr.flatten();
    }

    // Extracción del Genoma (Concatenación de todos los pesos y sesgos)
    std::vector<double> getGenome() const {
        std::vector<double> genome;
        for (const auto& w : weights) {
            auto v = w.flatten();
            genome.insert(genome.end(), v.begin(), v.end());
        }
        for (const auto& b : biases) {
            auto v = b.flatten();
            genome.insert(genome.end(), v.begin(), v.end());
        }
        return genome;
    }

    // Inyección del Genoma
    void setGenome(const std::vector<double>& genome) {
        size_t offset = 0;
        for (auto& w : weights) {
            w.fromVector(genome, offset);
        }
        for (auto& b : biases) {
            b.fromVector(genome, offset);
        }
        if (offset != genome.size()) {
            // El genoma provisto no coincide exactamente en tamaño
            throw std::invalid_argument("Tamaño de genoma incompatible con la topología.");
        }
    }

    // PERSISTENCIA: Guardar a archivo
    bool save(const std::string& filepath) const {
        std::ofstream file(filepath);
        if (!file.is_open()) return false;

        // Cabecera: Topología
        file << "TOPOLOGY: ";
        for (size_t n : topology) file << n << " ";
        file << "\n";
        
        // Cabecera: Activación
        file << "ACTIVATION: " << static_cast<int>(currentActivationType) << "\n";

        // Cuerpo: Pesos serializados
        std::vector<double> genome = getGenome();
        file << "WEIGHTS: ";
        file << std::fixed << std::setprecision(10);
        for (double w : genome) file << w << " ";
        file << "\n";

        return true;
    }

    // PERSISTENCIA: Cargar desde archivo (Método estático de fábrica)
    static NeuralNetwork load(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("No se pudo abrir el archivo.");
        }

        std::string line, label;
        std::vector<size_t> topo;
        int actId = 0; // Default Sigmoid

        // Leer Topología
        if (std::getline(file, line)) {
            std::stringstream ss(line);
            ss >> label; // "TOPOLOGY:"
            size_t val;
            while (ss >> val) topo.push_back(val);
        }

        if (topo.empty()) {
            throw std::runtime_error("Topología vacía o inválida.");
        }

        NeuralNetwork nn(topo);
        
        // Intentar leer Activación (si existe en el archivo)
        // Guardamos posición por si es un archivo antiguo
        std::streampos oldPos = file.tellg();
        if (std::getline(file, line)) {
             std::stringstream ss(line);
             ss >> label;
             if (label == "ACTIVATION:") {
                 ss >> actId;
                 nn.setActivationType(static_cast<ActivationType>(actId));
             } else {
                 // Si no es ACTIVATION, es WEIGHTS (archivo antiguo). Rebobinar.
                 file.seekg(oldPos);
             }
        }

        // Leer Pesos
        if (std::getline(file, line)) {
            std::stringstream ss(line);
            ss >> label; // "WEIGHTS:"
            std::vector<double> genome;
            double val;
            while (ss >> val) genome.push_back(val);
            nn.setGenome(genome);
        }

        return nn;
    }

    // Getter para topología (útil para el algoritmo genético)
    const std::vector<size_t>& getTopology() const { return topology; }
};

} // namespace evonet

#endif
