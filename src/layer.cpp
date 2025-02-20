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

    // std::cout << "ASDAS   " << u.size() << " " << A.size() << std::endl;

    MatrixXd next_u(u * A);

    A -= lambda * u.transpose() * x.transpose(); 
    b -= lambda * u.transpose();
    return next_u;
}

// BIG OPTIIZATION HERE - mutiply not matricies, but elementwise in case of sigmoid/relu
NonLinearLayer::NonLinearLayer(DefaultFunctions f) {
    if (f == DefaultFunctions::Sigmoid) {
        f_ = [](const VectorXd& x) { return Sigmoid(x); };
        b_ = [](const VectorXd& x, const MatrixXd& u) {
            if (u.size() != x.size()) {
                throw std::logic_error("NonLinearLayer wrong Backward sizes");
            }
            
            // auto u_ = u.cwiseProduct(SigmoidDeriv(x).transpose());
            MatrixXd d = SigmoidDeriv(x);
            // std::cout << u.cols() << "  " << u.rows() << std::endl;
            // std::cout << d.cols() << "  " << d.rows() << std::endl;
            MatrixXd next_u = u.cwiseProduct(SigmoidDeriv(x));
            // std::cout << u__.cols() << "  " << u__.rows() << std::endl;
            
            // std::cout << MatrixXd(u.array() * SigmoidDeriv(x).array()).transpose().cols() << ' ';
            // std::cout << MatrixXd(u.array()  * SigmoidDeriv(x).array()).transpose().rows() << std::endl;
            // u.cwiseProduct(SigmoidDeriv(x));
            return next_u;
        };
    } else if (f == DefaultFunctions::ReLU) {
        f_ = [](const VectorXd& x) { return ReLU(x); };
        b_ = [](const VectorXd& x, const MatrixXd& u) {
            if (u.size() != x.size()) {
                throw std::logic_error("NonLinearLayer wrong Backward sizes");
            }

            return MatrixXd(u * ReLUDeriv(x));
        };
    } else if (f == DefaultFunctions::Softmax) {
        f_ = [](const VectorXd& x) { return Softmax(x); };
        b_ = [](const VectorXd& x, const MatrixXd& u) {
            if (u.size() != x.size()) {
                throw std::logic_error("NonLinearLayer wrong Backward sizes");
            }

            return MatrixXd(u * SoftmaxDeriv(x));
        };
    } else if (f == DefaultFunctions::LeakyReLU) {
        f_ = [](const VectorXd& x) { return LeakyReLU(x); };
        b_ = [](const VectorXd& x, const MatrixXd& u) {
            if (u.size() != x.size()) {
                throw std::logic_error("NonLinearLayer wrong Backward sizes");
            }

            return MatrixXd(u * LeakyReLUDeriv(x));
        };
    }
}

VectorXd NonLinearLayer::Forward(const VectorXd& x) const {
    return f_(x); 
}

MatrixXd NonLinearLayer::Backward(const VectorXd& x, const MatrixXd& u, long double _) {
    return b_(x, u);
}

VectorXd NonLinearLayer::Sigmoid(const VectorXd& x) {
    return (1.0 / (1.0 + (-x.array()).exp())).matrix();
}

MatrixXd NonLinearLayer::SigmoidDeriv(const VectorXd& x) {
    return MatrixXd(Sigmoid(x).array() * (-Sigmoid(x).array() + 1)).transpose();
}

VectorXd NonLinearLayer::ReLU(const VectorXd& x) {
    return (x.array().max(0)).matrix();
}

MatrixXd NonLinearLayer::ReLUDeriv(const VectorXd& x) {
    return (0.5 + (x.array().sign())/2.0).matrix().asDiagonal();
}

VectorXd NonLinearLayer::Softmax(const VectorXd& x) {
    double m = x.maxCoeff();  // to prevent overflow
    VectorXd exp_x = (x.array() - m).exp();
    double exp_sum = exp_x.sum();
    return exp_x / exp_sum;
}

MatrixXd NonLinearLayer::SoftmaxDeriv(const VectorXd& x) {
    VectorXd s = Softmax(x);

    MatrixXd d = -s * s.transpose(); // -s_i * s_j
    d.diagonal() += s;  // -s_i * s_j + (i == j) * s_i

    return d;
}

VectorXd NonLinearLayer::LeakyReLU(const VectorXd& x) {
    double alpha = 0.01;
    return x.array().max(alpha * x.array()).matrix();
}

// not exactly deriv
MatrixXd NonLinearLayer::LeakyReLUDeriv(const VectorXd& x) {
    return (0.01 + 0.5 + (x.array().sign())/2.0).matrix().asDiagonal();
}

};  // namespace NeuralNetwork