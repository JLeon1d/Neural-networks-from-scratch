#include "layer.h"
#include "Eigen/Core"
#include "LinearAlgebra.h"
#include "gradient.h"

namespace NeuralNetwork {

LinearLayer::LinearLayer(size_t in_size, size_t out_size)
    : A_(Matrix::Random(out_size, in_size)), b_(Vector::Random(out_size)) {
}

Vector LinearLayer::Forward(const Vector& x) const {
    assert(x.size() == A_.cols() && "Wrong input vector size");

    return A_ * x + b_;
}

Matrix LinearLayer::Backward(const Vector& x, const RowVector& u, /* const */ GradientFunction& gf, double lambda) {
    assert(u.rows() == 1 && u.cols() == A_.rows() && "Wrong gradient vector size");

    Matrix next_u(u * A_);
    auto gradients = gf->operator()(x, u, lambda);
    A_ += gradients.first;
    b_ += gradients.second;

    return next_u;
}

/*
NonLinearLayer::NonLinearLayer(DefaultFunctions f) {
    if (f == DefaultFunctions::Sigmoid) {
        forward_ = std::move(Sigmoid);
        backward_ = [](const Vector& x, const RowVector& u) {
            assert(u.size() == x.size());

            RowVector next_u = u.cwiseProduct(SigmoidDeriv(x));
            return next_u;
        };
    } else if (f == DefaultFunctions::ReLU) {
        forward_ = std::move(ReLU);
        backward_ = [](const Vector& x, const RowVector& u) {
            assert(u.size() == x.size());

            return RowVector(u * ReLUDeriv(x));
        };
    } else if (f == DefaultFunctions::Softmax) {
        forward_ = std::move(Softmax);
        backward_ = [](const Vector& x, const RowVector& u) {
            assert(u.size() == x.size());

            return RowVector(u * SoftmaxDeriv(x));
        };
    } else if (f == DefaultFunctions::LeakyReLU) {
        forward_ = std::move(LeakyReLU);
        backward_ = [](const Vector& x, const RowVector& u) {
            assert(u.size() == x.size());

            return RowVector(u * LeakyReLUDeriv(x));
        };
    }
}
*/

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

Matrix NonLinearLayer::Backward(const Vector& x, const RowVector& u, /* const */ GradientFunction&, double) {
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
