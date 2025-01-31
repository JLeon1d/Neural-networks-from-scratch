#include "loss.h"
#include <stdexcept>

namespace NeuralNetwork {

long double LossFunction::Calculate(const VectorXd& z, const VectorXd& y) {
    if (z.size() != y.size()) {
        throw std::runtime_error("Vectors have different sizes");
    }

    return (z - y).array().square().mean();
}

MatrixXd LossFunction::Gradient(const VectorXd& z, const VectorXd& y) {
    return ((z - y).array() * 2.0/z.size()).matrix().transpose();
}

}; // namespace NeuralNetwork