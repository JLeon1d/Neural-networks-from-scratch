#include <stdexcept>
#include "layer.h"
#include "Eigen/Core"

#include <iostream>

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

// BIG OPTIIZATION HERE - mutiply not matricies, but elementwise in case of sigmoid/relu
NonLinearLayer::NonLinearLayer(DefaultFunctions f) {
    if (f == DefaultFunctions::Sigmoid) {
        f_ = [](const VectorXd& x) { return Sigmoid(x); };
        b_ = [](const VectorXd& x, const MatrixXd& u, long double _) {
            if (u.size() != x.size()) {
                throw std::logic_error("NonLinearLayer wrong Backward sizes");
            }

            return MatrixXd(u * SigmoidDeriv(x));
        };
    } else if (f == DefaultFunctions::ReLU) {
        f_ = [](const VectorXd& x) { return ReLU(x); };
        b_ = [](const VectorXd& x, const MatrixXd& u, long double _) {
            if (u.size() != x.size()) {
                throw std::logic_error("NonLinearLayer wrong Backward sizes");
            }

            std::cout << "RELUDeriv" << std::endl;
            std::cout << "x max/min: " << x.maxCoeff() << ' ' << x.minCoeff() << std::endl;
            std::cout << "u * relu_deriv(x) max/min: " << MatrixXd(u * ReLuDeriv(x)).maxCoeff() << ' ' << MatrixXd(u * ReLuDeriv(x)).minCoeff() << std::endl;

            return MatrixXd(u * ReLuDeriv(x));
        };
    } else if (f == DefaultFunctions::Softmax) {
        f_ = [](const VectorXd& x) { return Softmax(x); };
        b_ = [](const VectorXd& x, const MatrixXd& u, long double _) {
            if (u.size() != x.size()) {
                throw std::logic_error("NonLinearLayer wrong Backward sizes");
            }

            return MatrixXd(u * SoftmaxDeriv(x));
        };
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

MatrixXd NonLinearLayer::SigmoidDeriv(const VectorXd& x) {
    return VectorXd(Sigmoid(x).array() * (-Sigmoid(x).array() + 1).array()).asDiagonal();
}

VectorXd NonLinearLayer::ReLU(const VectorXd& x) {
    return (x.array().max(0)).matrix();
}

MatrixXd NonLinearLayer::ReLuDeriv(const VectorXd& x) {
    return (0.5 + (x.array().sign())/2.0).matrix().asDiagonal();
}

VectorXd NonLinearLayer::Softmax(const VectorXd& x) {
    double m = x.maxCoeff();  // to prevent overflow
    VectorXd exp_x = (x.array() - m).exp();
    double exp_sum = exp_x.sum();
    return exp_x / exp_sum;
}

MatrixXd NonLinearLayer::SoftmaxDeriv(const VectorXd& x) {
    int n = x.size();
    VectorXd s = NonLinearLayer::Softmax(x);

    // Initialize Jacobian matrix
    MatrixXd jacobian(n, n);

    // Compute Jacobian using the formula: d(s_i)/d(x_j) = s_i * (d_ij - s_j)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            jacobian(i, j) = s(i) * ((i == j ? 1.0 : 0.0) - s(j));
        }
    }

    return jacobian;
}

};  // namespace NeuralNetwork