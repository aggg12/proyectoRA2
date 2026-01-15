#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <ctime>
#include <iomanip>
#include <ale_interface.hpp>
#include "RamExtractor.hpp"

#ifdef __USE_SDL
  #include <SDL.h>
#endif

// --- Hyperparameters ---
const double LEARNING_RATE = 0.001;
const double GAMMA = 0.99;
const double L2_REGULARIZATION = 0.0001; // +0.20 Regularization
const int HIDDEN_NEURONS = 16;
const int NUM_EPISODES = 5000;
const int VALIDATION_INTERVAL = 50; // +0.15 Validation

// --- Activation Functions (+0.10 Selection) ---
enum ActivationType { ACT_RELU, ACT_SIGMOID, ACT_TANH };
const ActivationType HIDDEN_ACTIVATION = ACT_RELU;

double activate(double x, ActivationType type) {
    switch(type) {
        case ACT_RELU: return (x > 0) ? x : 0.0;
        case ACT_SIGMOID: return 1.0 / (1.0 + std::exp(-x));
        case ACT_TANH: return std::tanh(x);
        default: return x;
    }
}

double activate_derivative(double x, ActivationType type) {
    // Note: 'x' here is usually the output of the activation function for Sigmoid/Tanh,
    // or the input/output for ReLU depending on implementation. 
    // Here we assume 'x' is the OUTPUT of the layer for Sigmoid/Tanh, and input for ReLU simplification.
    switch(type) {
        case ACT_RELU: return (x > 0) ? 1.0 : 0.0; // x here is input
        case ACT_SIGMOID: return x * (1.0 - x); // x is output
        case ACT_TANH: return 1.0 - x * x; // x is output
        default: return 1.0;
    }
}

#include <fstream>

//  Neural Network Class 
class NeuralNetwork {
private:
    int n_input, n_hidden, n_output;
    std::vector<std::vector<double>> w1; // Weights Input -> Hidden
    std::vector<std::vector<double>> w2; // Weights Hidden -> Output
    std::vector<double> b1; // Biases Hidden
    std::vector<double> b2; // Biases Output
    
    // Cache for backprop
    std::vector<double> hidden_output;
    std::vector<double> final_output;
    std::vector<double> hidden_input_cache; // For ReLU derivative

    std::mt19937 rng;

public:
    NeuralNetwork(int inputs, int hidden, int outputs) 
        : n_input(inputs), n_hidden(hidden), n_output(outputs) {
        
        rng.seed(std::time(0));
        std::normal_distribution<double> dist(0.0, 0.1);

        // Init W1 (Inputs x Hidden)
        w1.resize(n_input, std::vector<double>(n_hidden));
        b1.resize(n_hidden);
        for(int i=0; i<n_input; ++i)
            for(int j=0; j<n_hidden; ++j) w1[i][j] = dist(rng);
        for(int j=0; j<n_hidden; ++j) b1[j] = 0.0;

        // Init W2 (Hidden x Output)
        w2.resize(n_hidden, std::vector<double>(n_output));
        b2.resize(n_output);
        for(int j=0; j<n_hidden; ++j)
            for(int k=0; k<n_output; ++k) w2[j][k] = dist(rng);
        for(int k=0; k<n_output; ++k) b2[k] = 0.0;
        
        hidden_output.resize(n_hidden);
        final_output.resize(n_output);
        hidden_input_cache.resize(n_hidden);
    }

    void save_weights(const std::string& filename) {
        std::ofstream file(filename);
        if (file.is_open()) {
            // W1
            for(const auto& row : w1) {
                for(double w : row) file << w << " ";
                file << "\n";
            }
            // B1
            for(double b : b1) file << b << " ";
            file << "\n";
            
            // W2
            for(const auto& row : w2) {
                for(double w : row) file << w << " ";
                file << "\n";
            }
            // B2
            for(double b : b2) file << b << " ";
            file << "\n";

            file.close();
            std::cout << "Weights saved to " << filename << std::endl;
        }
    }
}
