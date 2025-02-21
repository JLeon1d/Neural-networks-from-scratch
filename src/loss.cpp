#include "loss.h"
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

double LossFunction::Loss(const VectorXd& predicted, const VectorXd& target) const {
    return loss_(predicted, target);
}

MatrixXd LossFunction::Gradient(const VectorXd& predicted, const VectorXd& target) const {
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

double LossFunction::CrossEntropyLoss(const VectorXd& predicted, const VectorXd& target) {
    Eigen::ArrayXd p_log = predicted.array().log();
    p_log *= target.array();
    return -p_log.sum();
}

MatrixXd LossFunction::CrossEntropyGradient(const VectorXd& predicted, const VectorXd& target) {
    return (predicted - target).transpose();
}

}; // namespace NeuralNetwork