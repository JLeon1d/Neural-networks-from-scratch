#include "optimizer.h"

namespace NeuralNetwork {

namespace Optimizers {

namespace Caches {

Empty::Empty(size_t x_size, size_t u_size) {
}

SimpleState::SimpleState(size_t x_size, size_t u_size) : A(Matrix::Zero(u_size, x_size)), b(Vector::Zero(u_size)) {
}

Adam::Adam(size_t x_size, size_t u_size)
    : A_linear_(Matrix::Zero(u_size, x_size)),
      A_square_(Matrix::Zero(u_size, x_size)),
      b_linear_(Vector::Zero(u_size)),
      b_square_(Vector::Zero(u_size)),
      alpha_linear_step_(1.0),
      alpha_square_step_(1.0) {
}

};  // namespace Caches

Gradients Classic::Optimize(Cache&, const Vector& x, const RowVector& u, double learning_rate) const {
    return {-learning_rate * u.transpose() * x.transpose(), -learning_rate * u.transpose()};
}

Momentum::Momentum(double alpha) : alpha_(alpha) {
}

Gradients Momentum::Optimize(Cache& cache, const Vector& x, const RowVector& u, double learning_rate) const {
    assert(std::holds_alternative<RequiredCacheType>(cache) && "Wrong type of cache");
    auto m_cache = std::get<RequiredCacheType>(cache);

    m_cache.A = alpha_ * m_cache.A + learning_rate * u.transpose() * x.transpose();
    m_cache.b = alpha_ * m_cache.b + learning_rate * u.transpose();

    return {-m_cache.A, -m_cache.b};
}

Gradients AdaGrad::Optimize(Cache& cache, const Vector& x, const RowVector& u, double learning_rate) const {
    assert(std::holds_alternative<RequiredCacheType>(cache) && "Wrong type of cache");
    auto a_cache = std::get<RequiredCacheType>(cache);

    Matrix A_grad(u.transpose() * x.transpose());
    Vector b_grad(u.transpose());

    a_cache.A += A_grad.cwiseProduct(A_grad);
    a_cache.b += b_grad.cwiseProduct(b_grad);

    return {-learning_rate * A_grad.cwiseProduct((a_cache.A.array() + epsilon).sqrt().inverse().matrix()),
            -learning_rate * b_grad.cwiseProduct((a_cache.b.array() + epsilon).sqrt().inverse().matrix())};
}

RMSProp::RMSProp(double alpha) : alpha_(alpha) {
}

Gradients RMSProp::Optimize(Cache& cache, const Vector& x, const RowVector& u, double learning_rate) const {
    assert(std::holds_alternative<RequiredCacheType>(cache) && "Wrong type of cache");
    auto r_cache = std::get<RequiredCacheType>(cache);

    Matrix A_grad(u.transpose() * x.transpose());
    Vector b_grad(u.transpose());

    r_cache.A = alpha_ * r_cache.A + (1.0 - alpha_) * A_grad.cwiseProduct(A_grad);
    r_cache.b = alpha_ * r_cache.b + (1.0 - alpha_) * b_grad.cwiseProduct(b_grad);

    return {-learning_rate * A_grad.cwiseProduct((r_cache.A.array() + epsilon).sqrt().inverse().matrix()),
            -learning_rate * b_grad.cwiseProduct((r_cache.b.array() + epsilon).sqrt().inverse().matrix())};
}

Adam::Adam(double alpha_linear, double alpha_square) : alpha_linear_(alpha_linear), alpha_square_(alpha_square) {
}

Gradients Adam::Optimize(Cache& cache, const Vector& x, const RowVector& u, double learning_rate) const {
    assert(std::holds_alternative<RequiredCacheType>(cache) && "Wrong type of cache");
    auto a_cache = std::get<RequiredCacheType>(cache);

    Matrix A_grad(u.transpose() * x.transpose());
    Vector b_grad(u.transpose());

    a_cache.A_linear_ = alpha_linear_ * a_cache.A_linear_ + (1.0 - alpha_linear_) * A_grad;
    a_cache.b_linear_ = alpha_linear_ * a_cache.b_linear_ + (1.0 - alpha_linear_) * b_grad;

    a_cache.A_square_ = alpha_square_ * a_cache.A_square_ + (1.0 - alpha_square_) * A_grad.cwiseProduct(A_grad);
    a_cache.b_square_ = alpha_square_ * a_cache.b_square_ + (1.0 - alpha_square_) * b_grad.cwiseProduct(b_grad);

    a_cache.alpha_linear_step_ *= alpha_linear_;
    a_cache.alpha_square_step_ *= alpha_square_;

    Matrix A_m(((a_cache.A_square_.array() + epsilon) / (1.0 - a_cache.alpha_square_step_)).sqrt().inverse().matrix());
    Matrix b_m(((a_cache.b_square_.array() + epsilon) / (1.0 - a_cache.alpha_square_step_)).sqrt().inverse().matrix());

    return {
        -learning_rate * (a_cache.A_linear_.array() / (1.0 - a_cache.alpha_linear_step_)).matrix().cwiseProduct(A_m),
        -learning_rate * (a_cache.b_linear_.array() / (1.0 - a_cache.alpha_linear_step_)).matrix().cwiseProduct(b_m)};
}

};  // namespace Optimizers

};  // namespace NeuralNetwork
