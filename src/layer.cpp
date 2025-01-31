#include "layer.h"
#include <stdexcept>

namespace NeuralNetwork {

LinearLayer::LinearLayer(size_t in_size, size_t out_size)
    : in_size_(in_size), out_size_(out_size), A(MatrixXd::Random(out_size, in_size)),
      b(VectorXd::Random(out_size)) {
}

VectorXd LinearLayer::Forward(const VectorXd& x) const {
    if (x.size() != in_size_) {
        throw std::runtime_error("Wrong forward vector size");
    }

    return A * x + b;
}

MatrixXd LinearLayer::Backward(const VectorXd& x, const MatrixXd& u, long double lambda) {
    if (u.rows() != 1 or u.cols() != out_size_) {
        throw std::runtime_error("Wrong backward vector size");
    }

    MatrixXd next_u(u * A);
    A -= lambda * u.transpose() * x.transpose(); 
    b -= lambda * u.transpose();
    return next_u;
}

NonLinearLayer::NonLinearLayer(DefaultFunctions f) {
    if (f == DefaultFunctions::Sigmoid) {
        f_ = [](const VectorXd& x) { return Sigmoid(x); };
        b_ = [](const VectorXd& x, const VectorXd& u, long double _) { return u * SigmoidDeriv(x); };
    } else if (f == DefaultFunctions::ReLU) {
        f_ = [](const VectorXd& x) { return ReLU(x); };
        b_ = [](const VectorXd& x, const VectorXd& u, long double _) { return u * ReLuDeriv(x); };
    }
}

VectorXd NonLinearLayer::Forward(const VectorXd& x) const {
    return f_(x); 
}

MatrixXd NonLinearLayer::Backward(const VectorXd& x, const MatrixXd& u, long double lambda) {
    return b_(x, u, lambda);
}

VectorXd NonLinearLayer::Sigmoid(const VectorXd& x) {
    return (1.0 / (1.0 + (-x.array()).exp())).matrix();
}

VectorXd NonLinearLayer::SigmoidDeriv(const VectorXd& x) {
    return Sigmoid(x).array() * (-Sigmoid(x).array() + 1).array();
}

VectorXd NonLinearLayer::ReLU(const VectorXd& x) {
    return (x.array().max(0)).matrix();
}

VectorXd NonLinearLayer::ReLuDeriv(const VectorXd& x) {
    return (0.5 + (x.array().sign())/2.0).matrix();
}

};  // namespace NeuralNetwork