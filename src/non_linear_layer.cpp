#include "non_linear_layer.h"

namespace NeuralNetwork {

NonLinearLayer::NonLinearLayer(ActivationType type)
    : forward_(ActivationSelector::forward(type)), backward_(ActivationSelector::backward(type)) {
}

NonLinearLayer::NonLinearLayer(ForwardType forward, BackwardType backward)
    : forward_(std::move(forward)), backward_(std::move(backward)) {
}

Vector NonLinearLayer::Forward(const Vector& x) const {
    assert(forward_);
    return forward_(x);
}

Matrix NonLinearLayer::Backward(const Vector& x, const RowVector& u, const Optimizer&, Optimizers::Cache&, double) {
    assert(backward_);
    return backward_(x, u);
}

/* work in progress
Vector NonLinearLayer::LeakyReLU(const Vector& x) {
    double alpha = 0.01;
    return x.array().max(alpha * x.array()).matrix();
}

// not exactly deriv
Matrix NonLinearLayer::LeakyReLUDeriv(const Vector& x) {
    return (0.01 + 0.5 + (x.array().sign()) / 2.0).matrix().asDiagonal();
}
*/

};  // namespace NeuralNetwork
