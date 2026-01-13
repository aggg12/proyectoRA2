//probado 2/1/26 alex ilyas
#include "iostream"
#include <vector>
#include <cmath>
#include <algorithm>
#include <ctime>
#include <random>
#include <ionmanip>
#include <ale_interface.hpp>
#include "RamExtractor.hpp"

#ifdef __USE_SDL
    #include <SDL.h>
#endif

//hiperparamteros
const double ALPHA = 0.01;
const double GAMMA = 0.99;
const double EPSILON_START = 1.0;
const double EPSILON_END = 0.1;
const int EPSILON_DECAY_STEPS = 1000000; 
const int NUM_EPISODES = 5000;

enum ActionIndex { IDX_NOOP = 0, IDX_UP = 1, IDX_LEFT = 2, IDX_RIGHT = 3, IDX_RIGHTFIRE = 4, IDX_LEFTFIRE = 5 };
const int NUM_ACTIONS = 6;
const int NUM_FEATURES = 12;

#include <fstream>

class Perceptron {
public:
    std::vector<std::vector<double>> weights;
    std::mt19937 rng;

    Perceptron(){
        rng.seed(std::time(0));
        weights.resize(NUM_FEATURES, std::vector<double>(NUM_ACTIONS));
        std::uniform_real_distribution<double> dist(-0.01, 0.01);
        for(int i=0; i<NUM_FEATURES; ++i) {
            for(int a=0; a<NUM_ACTIONS; ++a) weights[i][a] = dist(rng);
        }
    }//04-01-26 alex ilyas prueba miguel ok hasta abajo

    void save_weights(const std::string& filename) {
        std::ofstream file(filename);
        if(file.is_open()) {
            for(const auto& row : weights) {
                for(double w : row) file << w << " ";
                file << "\n";
            }
            file.close();
            std::cout << "Weights saved to " << filename << std::endl;
        }
    }

    void load_weights(const std::string& filename) {
        std::ifstream file(filename);
        if(file.is_open()) {
            for(auto& row : weights) {
                for(double& w : row) file >> w;
            }
            file.close();
            std::cout << "Weights loaded from " << filename << std::endl;
        } else {
            std::cout << "Could not load weights from " << filename << std::endl;
        }
    }

    double activation(const std::vector<double>& features, int action_neuron_idx) {
        double output = 0;
        for(int i=0; i<NUM_FEATURES; ++i) output += features[i] * weights[i][action_neuron_idx];
        return output;
    }

    int predict(const std::vector<double>& features, double epsilon) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        if (dist(rng) < epsilon) return std::uniform_int_distribution<int>(0, NUM_ACTIONS - 1)(rng);
        int best_action = 0;
        double max_val = -1e9;
        for(int a=0; a<NUM_ACTIONS; ++a) {
            double val = activation(features, a);
            if(val > max_val) { max_val = val; best_action = a; }
        }
        return best_action;
    }

    double max_activation(const std::vector<double>& features) {
        double max_val = -1e9;
        for(int a=0; a<NUM_ACTIONS; ++a) {
            double val = activation(features, a);
            if(val > max_val) max_val = val;
        }
        return max_val;
    }

