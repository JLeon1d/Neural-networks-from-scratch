#include "gradient.h"
#include "Eigen/Core"
#include <utility>

namespace NeuralNetwork {

GradientPair ClassicGradient::operator()(const Vector& x, const RowVector& u, double learning_rate) {
    return std::make_pair(-learning_rate * u.transpose() * x.transpose(), -learning_rate * u.transpose());
}

MomentumGradient::MomentumGradient(size_t x_size, size_t u_size, double alpha)
    : momentum_A_(Matrix::Zero(u_size, x_size)), momentum_b_(Vector::Zero(u_size)), alpha_(alpha) {
}

GradientPair MomentumGradient::operator()(const Vector& x, const RowVector& u, double learning_rate) {
    momentum_A_ = alpha_ * momentum_A_ + learning_rate * u.transpose() * x.transpose();
    momentum_b_ = alpha_ * momentum_b_ + learning_rate * u.transpose();

    return std::make_pair(-momentum_A_, -momentum_b_);
}

AdaGradient::AdaGradient(size_t x_size, size_t u_size)
    : A_g_(Matrix::Zero(u_size, x_size)), b_g_(Vector::Zero(u_size)) {
}

GradientPair AdaGradient::operator()(const Vector& x, const RowVector& u, double learning_rate) {
    Matrix A_grad(u.transpose() * x.transpose());
    Vector b_grad(u.transpose());

    A_g_ += A_grad.cwiseProduct(A_grad);
    b_g_ += b_grad.cwiseProduct(b_grad);

    return std::make_pair(-learning_rate * A_grad.cwiseProduct((A_g_.array() + epsilon).sqrt().inverse().matrix()),
                          -learning_rate * b_grad.cwiseProduct((b_g_.array() + epsilon).sqrt().inverse().matrix()));
}

RMSPropGradient::RMSPropGradient(size_t x_size, size_t u_size, double alpha)
    : A_g_(Matrix::Zero(u_size, x_size)), b_g_(Vector::Zero(u_size)), alpha_(alpha) {
}

GradientPair RMSPropGradient::operator()(const Vector& x, const RowVector& u, double learning_rate) {
    Matrix A_grad(u.transpose() * x.transpose());
    Vector b_grad(u.transpose());

    A_g_ = alpha_ * A_g_ + (1.0 - alpha_) * A_grad.cwiseProduct(A_grad);
    b_g_ = alpha_ * b_g_ + (1.0 - alpha_) * b_grad.cwiseProduct(b_grad);

    return std::make_pair(-learning_rate * A_grad.cwiseProduct((A_g_.array() + epsilon).sqrt().inverse().matrix()),
                          -learning_rate * b_grad.cwiseProduct((b_g_.array() + epsilon).sqrt().inverse().matrix()));
}

AdamGradient::AdamGradient(size_t x_size, size_t u_size, double alpha_linear, double alpha_square)
    : A_linear_(Matrix::Zero(u_size, x_size)),
      A_square_(Matrix::Zero(u_size, x_size)),
      b_linear_(Vector::Zero(u_size)),
      b_square_(Vector::Zero(u_size)),
      alpha_linear_(alpha_linear),
      alpha_square_(alpha_square),
      alpha_linear_step_(1.0),
      alpha_square_step_(1.0) {
}

GradientPair AdamGradient::operator()(const Vector& x, const RowVector& u, double learning_rate) {
    Matrix A_grad(u.transpose() * x.transpose());
    Vector b_grad(u.transpose());

    A_linear_ = alpha_linear_ * A_linear_ + (1.0 - alpha_linear_) * A_grad;
    b_linear_ = alpha_linear_ * b_linear_ + (1.0 - alpha_linear_) * b_grad;

    A_square_ = alpha_square_ * A_square_ + (1.0 - alpha_square_) * A_grad.cwiseProduct(A_grad);
    b_square_ = alpha_square_ * b_square_ + (1.0 - alpha_square_) * b_grad.cwiseProduct(b_grad);

    alpha_linear_step_ *= alpha_linear_;
    alpha_square_step_ *= alpha_square_;

    Matrix A_m(((A_square_.array() + epsilon) / (1.0 - alpha_square_step_)).sqrt().inverse().matrix());
    Matrix b_m(((b_square_.array() + epsilon) / (1.0 - alpha_square_step_)).sqrt().inverse().matrix());

    return std::make_pair(-learning_rate * (A_linear_.array() / (1.0 - alpha_linear_step_)).matrix().cwiseProduct(A_m),
                          -learning_rate * (b_linear_.array() / (1.0 - alpha_linear_step_)).matrix().cwiseProduct(b_m));
}

};  // namespace NeuralNetwork
