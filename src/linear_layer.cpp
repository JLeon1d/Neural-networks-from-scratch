#include "linear_layer.h"

namespace NeuralNetwork {

LinearLayer::LinearLayer(In in_size, Out out_size)
    : A_(Matrix::Random(out_size, in_size)), b_(Vector::Random(out_size)) {
    assert(in_size >= 0 && out_size >= 0);
}

Vector LinearLayer::Forward(const Vector& x) const {
    assert(x.size() == A_.cols() && "Wrong input vector size");

    return A_ * x + b_;
}

Matrix LinearLayer::Backward(const Vector& x, const RowVector& u, const Optimizer& gf, Optimizers::Cache& cache,
                             double lambda) {
    assert(u.rows() == 1 && u.cols() == A_.rows() && "Wrong gradient vector size");

    Matrix next_u(u * A_);
    auto gradients = gf->Optimize(cache, x, u, lambda);
    A_ += gradients.A_grad;
    b_ += gradients.b_grad;

    return next_u;
}

LinearLayer::Weights LinearLayer::GetWeights() const {
    return Weights{A_, b_};
}

};  // namespace NeuralNetwork
