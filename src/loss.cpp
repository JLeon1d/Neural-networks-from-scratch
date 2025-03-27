#include "loss.h"

namespace NeuralNetwork {

LossFunction::LossFunction(LossType type) : loss_(LossSelector::loss(type)), gradient_(LossSelector::gradient(type)) {
}

LossFunction::LossFunction(LossFunctionType loss, GradienFunctionType gradient)
    : loss_(std::move(loss)), gradient_(std::move(gradient)) {
}

double LossFunction::Loss(const Vector& predicted, const Vector& target) const {
    assert(loss_);
    return loss_(predicted, target);
}

RowVector LossFunction::Gradient(const Vector& predicted, const Vector& target) const {
    assert(gradient_);
    return gradient_(predicted, target);
}

LossFunction LossFunction::Mse() {
    return LossFunction(LossType::Mse);
}

LossFunction LossFunction::CrossEntropy() {
    return LossFunction(LossType::CrossEntropy);
}

};  // namespace NeuralNetwork
