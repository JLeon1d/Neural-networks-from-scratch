#include "loss.h"
#include <stdexcept>

namespace NeuralNetwork {

LossFunction::LossFunction(LossFunctionType loss, GradienFunctionType gradient)
    : loss_(std::move(loss)), gradient_(std::move(gradient)) {
}

double LossFunction::Loss(const VectorXd& predicted, const VectorXd& target) {
    return loss_(predicted, target);
}

MatrixXd LossFunction::Gradient(const VectorXd& predicted, const VectorXd& target) {
    return gradient_(predicted, target);
}

double LossFunction::MSELoss(const VectorXd& predicted, const VectorXd& target) {
    if (predicted.size() != target.size()) {
        throw std::runtime_error("Vectors have different sizes");
    }

    return (predicted - target).array().square().mean();
}

MatrixXd LossFunction::MSEGradient(const VectorXd& predicted, const VectorXd& target) {
    return ((predicted - target).array() * 2.0/predicted.size()).matrix().transpose();
}

}; // namespace NeuralNetwork