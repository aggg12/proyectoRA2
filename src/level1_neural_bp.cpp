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

// --- Neural Network Class ---
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

    void load_weights(const std::string& filename) {
        std::ifstream file(filename);
        if (file.is_open()) {
            for(auto& row : w1) for(double& w : row) file >> w;
            for(double& b : b1) file >> b;
            for(auto& row : w2) for(double& w : row) file >> w;
            for(double& b : b2) file >> b;
            file.close();
            std::cout << "Weights loaded from " << filename << std::endl;
        } else {
             std::cout << "Could not load weights from " << filename << std::endl;
        }
    }

    std::vector<double> forward(const std::vector<double>& inputs) {
        // Input -> Hidden
        for(int j=0; j<n_hidden; ++j) {
            double sum = b1[j];
            for(int i=0; i<n_input; ++i) {
                sum += inputs[i] * w1[i][j];
            }
            hidden_input_cache[j] = sum; // Cache 'z' for ReLU
            hidden_output[j] = activate(sum, HIDDEN_ACTIVATION);
        }

        // Hidden -> Output (Linear activation for Q-values)
        for(int k=0; k<n_output; ++k) {
            double sum = b2[k];
            for(int j=0; j<n_hidden; ++j) {
                sum += hidden_output[j] * w2[j][k];
            }
            final_output[k] = sum;
        }
        return final_output;
    }

    void train(const std::vector<double>& inputs, int action, double target) {
        // 1. Forward Pass (Update cache)
        forward(inputs);

        // 2. Backpropagation
        // Output Layer Error (Target - Output)
        double output_error = final_output[action] - target; 

        // Calculate Hidden Error (BEFORE updating weights!)
        std::vector<double> hidden_error(n_hidden, 0.0);
        for(int j=0; j<n_hidden; ++j) {
            hidden_error[j] = output_error * w2[j][action];
        }

        // Update Output Weights (W2, B2)
        for(int j=0; j<n_hidden; ++j) {
            double grad = output_error * hidden_output[j];
            // L2 Regularization
            w2[j][action] -= LEARNING_RATE * (grad + L2_REGULARIZATION * w2[j][action]);
        }
        b2[action] -= LEARNING_RATE * output_error;

        // Update Hidden Weights (W1, B1)
        for(int j=0; j<n_hidden; ++j) {
            // Derivative of activation
            double derivative = 0;
            if (HIDDEN_ACTIVATION == ACT_RELU) 
                derivative = activate_derivative(hidden_input_cache[j], ACT_RELU);
            else 
                derivative = activate_derivative(hidden_output[j], HIDDEN_ACTIVATION);

            double delta = hidden_error[j] * derivative;

            for(int i=0; i<n_input; ++i) {
                double grad = delta * inputs[i];
                w1[i][j] -= LEARNING_RATE * (grad + L2_REGULARIZATION * w1[i][j]);
            }
            b1[j] -= LEARNING_RATE * delta;
        }
    }
};

// --- Features (Reuse from Level 0) ---
const int NUM_FEATURES = 12;
std::vector<double> extract_features(const GameState& state) {
    std::vector<double> f(NUM_FEATURES, 0.0);
    f[0] = 1.0; // Bias
    f[1] = (double)(state.cannonX - 4) / 16.0;

    // Enemy Relative Positions
    f[2] = (state.enemy1X != -1) ? (double)(state.enemy1X - state.cannonX) / 32.0 : 0.0;
    f[3] = (state.enemy2X != -1) ? (double)(state.enemy2X - state.cannonX) / 32.0 : 0.0;
    f[4] = (state.enemy3X != -1) ? (double)(state.enemy3X - state.cannonX) / 32.0 : 0.0;

    // Projectile Info
    f[5] = (double)state.projectileActive;
    if (state.projectileActive) {
        f[6] = (double)(state.projectileX - state.cannonX) / 32.0;
        f[7] = (double)state.projectileY / 255.0;
    } else {
        f[6] = 0.0;
        f[7] = 1.0; 
    }

    f[8] = (double)state.temperature / 14.0;
    f[9] = (double)state.levelType / 3.0;

    // Dodge Room
    f[10] = (double)(state.cannonX - 4) / 16.0; // Left Room
    f[11] = (double)(20 - state.cannonX) / 16.0; // Right Room

    return f;
}

// --- Main ---
int main(int argc, char** argv) {
    if (argc < 2) { 
        std::cerr << "Uso: ./level1 <rom_file> [load] [display]" << std::endl; 
        return 1; 
    }
    
    bool load_mode = false;
    bool display_mode = false;
    
    for(int i=2; i<argc; ++i) {
        std::string arg = argv[i];
        if (arg == "load") load_mode = true;
        if (arg == "display") display_mode = true;
    }
    
    ALEInterface ale;
    ale.setInt("random_seed", 123);
    ale.setBool("display_screen", display_mode); 
    ale.setBool("sound", display_mode);
    ale.setInt("frame_skip", 4);
    ale.loadROM(argv[1]);

    Action actions[] = { PLAYER_A_NOOP, PLAYER_A_UP, PLAYER_A_LEFT, PLAYER_A_RIGHT, PLAYER_A_RIGHTFIRE, PLAYER_A_LEFTFIRE };
    const int NUM_ACTIONS = 6;

    // Initialize MLP: 10 Inputs -> 16 Hidden (ReLU) -> 6 Outputs (Linear)
    NeuralNetwork nn(NUM_FEATURES, HIDDEN_NEURONS, NUM_ACTIONS);

    double epsilon = 1.0;
    int episodes_to_run = NUM_EPISODES;
    
    if (load_mode) {
        nn.load_weights("level1.weights");
        epsilon = 0.0;
        std::cout << "=== PLAY MODE (Loaded Weights) ===" << std::endl;
        episodes_to_run = 10;
    } else {
        std::cout << "=== LEVEL 1: NEURAL NETWORK BACKPROPAGATION ===" << std::endl;
        std::cout << "Structure: 10 -> " << HIDDEN_NEURONS << "(ReLU) -> 6(Linear)" << std::endl;
        std::cout << "Features: Backpropagation, L2 Regularization, Validation Sets" << std::endl;
    }

    for (int episode = 0; episode < episodes_to_run; ++episode) {
        
        // --- Validation Logic (+0.15) ---
        bool is_validation = (episode > 0) && (episode % VALIDATION_INTERVAL == 0);
        double current_epsilon = (is_validation || load_mode) ? 0.0 : epsilon; 
        
        ale.reset_game();
        GameState state = RamExtractor::extract(ale);
        std::vector<double> features = extract_features(state);
        
        double total_reward = 0;
        int lives = state.lives;
        int frames_still = 0;
        int last_cannonX = -1;

        int frame_count = 0;
        const int MAX_FRAMES = 6000;

        while (!ale.game_over() && frame_count < MAX_FRAMES) {
            // Forward Pass
            std::vector<double> q_values = nn.forward(features);
            
            // Epsilon-Greedy Selection
            int action_idx = 0;
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            static std::mt19937 rng(std::time(0));
            
            if (dist(rng) < current_epsilon) {
                action_idx = std::uniform_int_distribution<int>(0, NUM_ACTIONS - 1)(rng);
            } else {
                // Argmax
                double max_val = -1e9;
                for(int i=0; i<NUM_ACTIONS; ++i) {
                    if(q_values[i] > max_val) { max_val = q_values[i]; action_idx = i; }
                }
            }

            // Actuator
            Action ale_action = actions[action_idx];
            
            double reward = ale.act(ale_action);
            total_reward += reward;

            GameState next_state = RamExtractor::extract(ale);
            std::vector<double> next_features = extract_features(next_state);

            // Update trackers
            if (next_state.lives < lives) lives = next_state.lives;
            last_cannonX = next_state.cannonX;

            // Train only if NOT validation AND NOT load mode
            if (!is_validation && !load_mode) {
                // Heat Penalty
                double shaped_reward = reward;
                if (next_state.temperature > 10) shaped_reward -= 1.0;

                // Calculate Target (Q-Learning Bellman)
                std::vector<double> next_q = nn.forward(next_features);
                double max_next_q = -1e9;
                for(double q : next_q) if(q > max_next_q) max_next_q = q;
                
                double target = shaped_reward;
                if (!ale.game_over()) target += GAMMA * max_next_q;

                // Backpropagation
                nn.train(features, action_idx, target);
            }

            state = next_state;
            features = next_features;
            frame_count++;
        }

        // Decay Epsilon (Slower decay: 0.999)
        if (!is_validation && !load_mode && epsilon > 0.1) epsilon *= 0.999;

        // Logging
        if (load_mode || is_validation) {
             std::cout << (load_mode ? "[PLAY] Ep " : "[VALIDATION] Ep ") << episode << " | Score: " << total_reward << std::endl;
        } else if ((episode + 1) % 10 == 0) {
             std::cout << "Ep: " << (episode + 1) << " | Score: " << total_reward << " | Eps: " << epsilon << std::endl;
        }
    }
    
    if (!load_mode) {
        nn.save_weights("level1.weights");
    }
    
    return 0;
}
