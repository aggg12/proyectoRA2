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

    void train(const std::vector<double>& inputs, int action, double reward, const std::vector<double>& next_inputs, bool done) {
        double current_output = activation(inputs, action);
        double target = reward;
        if (!done) target += GAMMA * max_activation(next_inputs);
        double error = target - current_output;
        for(int i=0; i<NUM_FEATURES; ++i) weights[i][action] += ALPHA * error * inputs[i];
    }
};

std::vector<double> extract_features(const GameState& state) {
    std::vector<double> f(NUM_FEATURES, 0.0);
    f[0] = 1.0; // Bias
    // Player position normalized in its real range [4, 20]
    f[1] = (double)(state.cannonX - 4) / 16.0;

    // Enemy Relative Positions (can be outside player range, use 32.0 for safety)
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

    // Dodge Room (Space to move) - Precise range 4 to 20
    f[10] = (double)(state.cannonX - 4) / 16.0; // Room Left (0 at edge)
    f[11] = (double)(20 - state.cannonX) / 16.0; // Room Right (0 at edge)

    return f;
}

int main(int argc, char** argv){
    if (argc < 2) { 
        std::cerr << "Uso: ./level0 <rom_file> [load] [display]" << std::endl; 
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

    Perceptron brain;
    Action actions[] = { PLAYER_A_NOOP, PLAYER_A_UP, PLAYER_A_LEFT, PLAYER_A_RIGHT, PLAYER_A_RIGHTFIRE, PLAYER_A_LEFTFIRE };
    
    double epsilon = EPSILON_START;
    int episodes_to_run = NUM_EPISODES;

    if (load_mode) {
        brain.load_weights("level0.weights");
        epsilon = 0.0; // Play perfectly
        std::cout << "=== PLAY MODE (Loaded Weights) ===" << std::endl;
        episodes_to_run = 10; // Just show a few games
    } else {
        std::cout << "=== LEVEL 0: FAST PERCEPTRON TRAINING ===" << std::endl;
        std::cout << "Training for " << NUM_EPISODES << " episodes..." << std::endl;
    }

    for (int episode = 0; episode < episodes_to_run; ++episode) {
        ale.reset_game();
        GameState state = RamExtractor::extract(ale);
        std::vector<double> features = extract_features(state);
        double episode_score = 0;
        int lives = state.lives;
        int last_cannonX = -1;

        int frame_count = 0;
        const int MAX_FRAMES = 6000;

        while (!ale.game_over() && frame_count < MAX_FRAMES) {
            int action_idx = brain.predict(features, epsilon);
            Action ale_action = actions[action_idx];
            
            // Pure AI: Direct Action
            double reward = ale.act(ale_action);
            episode_score += reward;

            GameState next_state = RamExtractor::extract(ale);
            std::vector<double> next_features = extract_features(next_state);

            // Train only if NOT in load/play mode
            if (!load_mode) {
                // Update trackers
                if (next_state.lives < lives) lives = next_state.lives;
                last_cannonX = next_state.cannonX;

                // Heat Penalty
                double shaped_reward = reward;
                if (next_state.temperature > 10) shaped_reward -= 1.0;

                // Train on shaped reward
                brain.train(features, action_idx, shaped_reward, next_features, ale.game_over());
                
                if (epsilon > EPSILON_END) epsilon -= (EPSILON_START - EPSILON_END) / EPSILON_DECAY_STEPS;
            }

            state = next_state; features = next_features;
            frame_count++;
        }

        std::cout << "Ep: " << (episode + 1) 
                  << " | Score: " << (int)episode_score 
                  << " | Eps: " << std::fixed << std::setprecision(2) << epsilon << std::endl;
    }

    if (!load_mode) {
        brain.save_weights("level0.weights");
    }
    
    return 0;
}


