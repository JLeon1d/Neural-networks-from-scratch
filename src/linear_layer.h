#pragma once

#include "LinearAlgebra.h"
#include "optimizer.h"
#include "layer.h"

namespace NeuralNetwork {

enum In : size_t;
enum Out : size_t;
class LinearLayer {
public:
    LinearLayer(In in_size, Out out_size);

    ~LinearLayer() = default;

    Vector Forward(const Vector& x) const;
    Matrix Backward(const Vector& x, const RowVector& u, const Optimizer& gf, Optimizers::Cache& cache, double lambda);

    LayerWeights GetWeights() const;

private:
    Matrix A_;
    Vector b_;
};

};  // namespace NeuralNetwork
