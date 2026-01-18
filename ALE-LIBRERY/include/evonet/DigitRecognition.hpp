#ifndef EVONET_DIGIT_RECOGNITION_HPP
#define EVONET_DIGIT_RECOGNITION_HPP

#include <vector>
#include <iostream>

namespace evonet {

class DigitRecognition {
public:
    struct Sample {
        std::vector<double> pixels; // 15 inputs (3x5)
        std::vector<double> targets; // 10 outputs (One-hot)
        int label;
    };

    std::vector<Sample> dataset;

    DigitRecognition() {
        // Definimos los 'sprites' de 5x7 para números del 0 al 9 (Alta Resolución)
        // 1 = Pixel pintado, 0 = Pixel vacio
        std::vector<std::vector<double>> rawDigits = {
            // 0
            {0,1,1,1,0,
             1,0,0,0,1,
             1,0,0,0,1,
             1,0,0,0,1,
             1,0,0,0,1,
             1,0,0,0,1,
             0,1,1,1,0},
            // 1
            {0,0,1,0,0,
             0,1,1,0,0,
             0,0,1,0,0,
             0,0,1,0,0,
             0,0,1,0,0,
             0,0,1,0,0,
             0,1,1,1,0},
            // 2
            {0,1,1,1,0,
             1,0,0,0,1,
             0,0,0,0,1,
             0,0,0,1,0,
             0,0,1,0,0,
             0,1,0,0,0,
             1,1,1,1,1},
            // 3
            {1,1,1,1,1,
             0,0,0,1,0,
             0,0,1,0,0,
             0,0,0,1,0,
             0,0,0,0,1,
             1,0,0,0,1,
             0,1,1,1,0},
            // 4
            {0,0,0,1,0,
             0,0,1,1,0,
             0,1,0,1,0,
             1,0,0,1,0,
             1,1,1,1,1,
             0,0,0,1,0,
             0,0,0,1,0},
            // 5
            {1,1,1,1,1,
             1,0,0,0,0,
             1,1,1,1,0,
             0,0,0,0,1,
             0,0,0,0,1,
             1,0,0,0,1,
             0,1,1,1,0},
            // 6
            {0,1,1,1,0,
             1,0,0,0,0,
             1,0,0,0,0,
             1,1,1,1,0,
             1,0,0,0,1,
             1,0,0,0,1,
             0,1,1,1,0},
            // 7
            {1,1,1,1,1,
             0,0,0,0,1,
             0,0,0,1,0,
             0,0,1,0,0,
             0,1,0,0,0,
             0,1,0,0,0,
             0,1,0,0,0},
            // 8
            {0,1,1,1,0,
             1,0,0,0,1,
             1,0,0,0,1,
             0,1,1,1,0,
             1,0,0,0,1,
             1,0,0,0,1,
             0,1,1,1,0},
            // 9
            {0,1,1,1,0,
             1,0,0,0,1,
             1,0,0,0,1,
             0,1,1,1,1,
             0,0,0,0,1,
             0,0,0,1,0,
             0,1,1,0,0}
        };

        for (int i = 0; i < 10; ++i) {
            Sample s;
            s.pixels = rawDigits[i];
            s.label = i;
            
            // Output deseado: Todo ceros excepto el índice correcto
            s.targets = std::vector<double>(10, 0.0);
            s.targets[i] = 1.0;
            
            dataset.push_back(s);
        }
    }

    const std::vector<Sample>& getData() const {
        return dataset;
    }
};

} // namespace evonet

#endif
