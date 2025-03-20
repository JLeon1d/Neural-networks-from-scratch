#include "layer.h"
#include "Eigen/Core"
#include "LinearAlgebra.h"
#include "gradient.h"

#include <stdexcept>

namespace NeuralNetwork {

LinearLayer::LinearLayer(size_t in_size, size_t out_size)
    : in_size_(in_size), out_size_(out_size), A(Matrix::Random(out_size, in_size)),
      b(Vector::Random(out_size)) {
}

Vector LinearLayer::Forward(const Vector& x) const {
    if (x.size() != in_size_) {
        throw std::runtime_error("Wrong forward vector size");
    }

    return A * x + b;
}

Matrix LinearLayer::Backward(const Vector& x, const RowVector& u, GradientFunction& gf, long double lambda) {
    if (u.rows() != 1 or u.cols() != out_size_) {
        throw std::runtime_error("Wrong backward vector size");
    }

    Matrix next_u(u * A);
    auto gradients = gf->operator()(x, u, lambda);
    A += gradients.first;
    b += gradients.second;

    return next_u;
}

NonLinearLayer::NonLinearLayer(DefaultFunctions f) {
    if (f == DefaultFunctions::Sigmoid) {
        f_ = std::move(Sigmoid);
        b_ = [](const Vector& x, const Matrix& u) {
            if (u.size() != x.size()) {
                throw std::logic_error("NonLinearLayer wrong Backward sizes");
            }

            Matrix next_u = u.cwiseProduct(SigmoidDeriv(x));
            return next_u;
        };
    } else if (f == DefaultFunctions::ReLU) {
        f_ = std::move(ReLU);
        b_ = [](const Vector& x, const Matrix& u) {
            if (u.size() != x.size()) {
                throw std::logic_error("NonLinearLayer wrong Backward sizes");
            }

            return Matrix(u * ReLUDeriv(x));
        };
    } else if (f == DefaultFunctions::Softmax) {
        f_ = std::move(Softmax);
        b_ = [](const Vector& x, const Matrix& u) {
            if (u.size() != x.size()) {
                throw std::logic_error("NonLinearLayer wrong Backward sizes");
            }

            return Matrix(u * SoftmaxDeriv(x));
        };
    } else if (f == DefaultFunctions::LeakyReLU) {
        f_ = std::move(LeakyReLU);
        b_ = [](const Vector& x, const Matrix& u) {
            if (u.size() != x.size()) {
                throw std::logic_error("NonLinearLayer wrong Backward sizes");
            }

            return Matrix(u * LeakyReLUDeriv(x));
        };
    }
}

Vector NonLinearLayer::Forward(const Vector& x) const {
    return f_(x); 
}

Matrix NonLinearLayer::Backward(const Vector& x, const RowVector& u, GradientFunction& _, long double __) {
    return b_(x, u);
}

Vector NonLinearLayer::Sigmoid(const Vector& x) {
    return (1.0 / (1.0 + (-x.array()).exp())).matrix();
}

Matrix NonLinearLayer::SigmoidDeriv(const Vector& x) {
    return Matrix(Sigmoid(x).array() * (-Sigmoid(x).array() + 1)).transpose();
}

Vector NonLinearLayer::ReLU(const Vector& x) {
    return (x.array().max(0)).matrix();
}

Matrix NonLinearLayer::ReLUDeriv(const Vector& x) {
    return (0.5 + (x.array().sign())/2.0).matrix().asDiagonal();
}

Vector NonLinearLayer::Softmax(const Vector& x) {
    double m = x.maxCoeff();  // to prevent overflow
    Vector exp_x = (x.array() - m).exp();
    double exp_sum = exp_x.sum();
    return exp_x / exp_sum;
}

Matrix NonLinearLayer::SoftmaxDeriv(const Vector& x) {
    Vector s = Softmax(x);

    Matrix d = -s * s.transpose(); // -s_i * s_j
    d.diagonal() += s;  // -s_i * s_j + (i == j) * s_i

    return d;
}

Vector NonLinearLayer::LeakyReLU(const Vector& x) {
    double alpha = 0.01;
    return x.array().max(alpha * x.array()).matrix();
}

// not exactly deriv
Matrix NonLinearLayer::LeakyReLUDeriv(const Vector& x) {
    return (0.01 + 0.5 + (x.array().sign())/2.0).matrix().asDiagonal();
}

};  // namespace NeuralNetwork