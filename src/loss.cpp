#include "loss.h"
#include "LinearAlgebra.h"
#include <stdexcept>

namespace NeuralNetwork {

LossFunction::LossFunction(LossFunctionType loss, GradienFunctionType gradient)
    : loss_(std::move(loss)), gradient_(std::move(gradient)) {
}

LossFunction::LossFunction(LossFunction::Default func) {
    if (func == LossFunction::MSE) {
        loss_ = std::move(MSELoss);
        gradient_ = std::move(MSEGradient);
    } else if (func == LossFunction::CrossEntropy) {
        loss_ = std::move(CrossEntropyLoss);
        gradient_ = std::move(CrossEntropyGradient);
    }
}

double LossFunction::Loss(const Vector& predicted, const Vector& target) const {
    return loss_(predicted, target);
}

RowVector LossFunction::Gradient(const Vector& predicted, const Vector& target) const {
    return gradient_(predicted, target);
}

double LossFunction::MSELoss(const Vector& predicted, const Vector& target) {
    if (predicted.size() != target.size()) {
        throw std::runtime_error("Vectors have different sizes");
    }

    return (predicted - target).array().square().mean();
}

RowVector LossFunction::MSEGradient(const Vector& predicted, const Vector& target) {
    return ((predicted - target).array() * 2.0/predicted.size()).matrix().transpose();
}

double LossFunction::CrossEntropyLoss(const Vector& predicted, const Vector& target) {
    Eigen::ArrayXd p_log = predicted.array().log();
    p_log *= target.array();
    return -p_log.sum();
}

RowVector LossFunction::CrossEntropyGradient(const Vector& predicted, const Vector& target) {
    return (predicted - target).transpose();
}

}; // namespace NeuralNetwork