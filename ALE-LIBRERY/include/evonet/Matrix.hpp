/**
 * @file Matrix.hpp
 * @brief Implementación de operaciones matriciales básicas para redes neuronales.
 */
#ifndef EVONET_MATRIX_HPP
#define EVONET_MATRIX_HPP

#include <vector>
#include <iostream>
#include <random>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <iomanip>

namespace evonet {

class Matrix {
private:
    std::vector<std::vector<double>> data;
    size_t rows;
    size_t cols;

public:
    Matrix() : rows(0), cols(0) {}

    // Constructor
    Matrix(size_t r, size_t c, bool randomize = false) : rows(r), cols(c) {
        data.resize(rows, std::vector<double>(cols, 0.0));
        if (randomize) {
            // Inicialización Xavier/Glorot simplificada para Tanh/Sigmoid
            double limit = std::sqrt(6.0 / (rows + cols));
            applyRandom(-limit, limit);
        }
    }

    // Acceso a dimensiones
    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }

    // Acceso a datos (lectura/escritura)
    double& at(size_t r, size_t c) {
        if (r >= rows || c >= cols) {
            throw std::out_of_range("Índice de matriz fuera de rango");
        }
        return data[r][c];
    }

    const double& at(size_t r, size_t c) const {
        if (r >= rows || c >= cols) {
            throw std::out_of_range("Índice de matriz fuera de rango");
        }
        return data[r][c];
    }

    // Operación: Aleatorización
    void applyRandom(double minVal, double maxVal) {
        static thread_local std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<double> dis(minVal, maxVal);

        for (auto& row : data) {
            for (auto& val : row) {
                val = dis(gen);
            }
        }
    }

    // Operación: Multiplicación Matricial (Producto Punto)
    // O(rows * cols * other.cols)
    Matrix multiply(const Matrix& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Dimensiones incompatibles para multiplicación matricial.");
        }
        Matrix result(rows, other.cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t k = 0; k < cols; ++k) {
                // Optimización simple: si el valor es 0, saltar
                if (data[i][k] == 0.0) continue;
                for (size_t j = 0; j < other.cols; ++j) {
                    result.data[i][j] += data[i][k] * other.data[k][j];
                }
            }
        }
        return result;
    }

    // Operación: Suma elemento a elemento (para Bias)
    void add(const Matrix& other) {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Dimensiones incompatibles para suma.");
        }
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                data[i][j] += other.data[i][j];
            }
        }
    }

    // Operación: Mapeo de función (para Activaciones)
    Matrix map(const std::function<double(double)>& func) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.data[i][j] = func(data[i][j]);
            }
        }
        return result;
    }

    // Utilidad: Aplanar matriz a vector (Genotipo)
    std::vector<double> flatten() const {
        std::vector<double> vec;
        vec.reserve(rows * cols);
        for (const auto& row : data) {
            vec.insert(vec.end(), row.begin(), row.end());
        }
        return vec;
    }

    // Utilidad: Reconstruir desde vector
    void fromVector(const std::vector<double>& vec, size_t& offset) {
        if (vec.size() < offset + rows * cols) {
            throw std::out_of_range("Vector insuficiente para llenar matriz.");
        }
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                data[i][j] = vec[offset++];
            }
        }
    }

    // Debug
    void print() const {
        for (const auto& row : data) {
            for (double val : row) {
                std::cout << std::fixed << std::setprecision(4) << val << " ";
            }
            std::cout << "\n";
        }
    }
};

} // namespace evonet

#endif
