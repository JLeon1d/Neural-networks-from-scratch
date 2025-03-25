#pragma once

#include <cstdint>
#include "LinearAlgebra.h"
#include "gradient.h"

namespace NeuralNetwork {

enum In : int64_t;
enum Out : int64_t;
class LinearLayer {
public:
    LinearLayer(In in_size, Out out_size);

    ~LinearLayer() = default;

    Vector Forward(const Vector& x) const;
    Matrix Backward(const Vector& x, const RowVector& u, /* const */ GradientFunction& gf, double lambda);

private:
    Matrix A_;
    Vector b_;
};

};  // namespace NeuralNetwork
