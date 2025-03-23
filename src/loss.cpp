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
    assert(loss_);
    return loss_(predicted, target);
}

RowVector LossFunction::Gradient(const Vector& predicted, const Vector& target) const {
    assert(gradient_);
    return gradient_(predicted, target);
}

double LossFunction::MSELoss(const Vector& predicted, const Vector& target) {
    assert(predicted.size() == target.size());

    return (predicted - target).array().square().mean();
}

RowVector LossFunction::MSEGradient(const Vector& predicted, const Vector& target) {
    return ((predicted - target).array() * 2.0 / predicted.size()).matrix().transpose();
}

double LossFunction::CrossEntropyLoss(const Vector& predicted, const Vector& target) {
    auto p_log = predicted.array().log();
    return -(p_log * target.array()).sum();
}

RowVector LossFunction::CrossEntropyGradient(const Vector& predicted, const Vector& target) {
    return (predicted - target).transpose();
}

};  // namespace NeuralNetwork
