#include "non_linear_layer.h"
#include "LinearAlgebra.h"

namespace NeuralNetwork {

NonLinearLayer::NonLinearLayer(ActivationType type)
    : forward_(ActivationSelector::forward(type)), backward_(ActivationSelector::backward(type)) {
}

Vector NonLinearLayer::Forward(const Vector& x) const {
    assert(forward_);
    return forward_(x);
}

Matrix NonLinearLayer::Backward(const Vector& x, const RowVector& u, const Optimizer&, Optimizers::Cache&, double) {
    assert(backward_);
    return backward_(x, u);
}

NonLinearLayer NonLinearLayer::Sigmoid() {
    return NonLinearLayer(ActivationType::Sigmoid);
}

NonLinearLayer NonLinearLayer::ReLU() {
    return NonLinearLayer(ActivationType::ReLU);
}

NonLinearLayer NonLinearLayer::LeakyReLU(double alpha) {
    return NonLinearLayer(
        [alpha](const Vector& x) { return ActivationData<ActivationType::LeakyReLU>::forward(x, alpha); },
        [alpha](const Vector& x, const RowVector& u) {
            return ActivationData<ActivationType::LeakyReLU>::backward(x, u, alpha);
        });
}

NonLinearLayer NonLinearLayer::Softmax() {
    return NonLinearLayer(ActivationType::Softmax);
}

};  // namespace NeuralNetwork
