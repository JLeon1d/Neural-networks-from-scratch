#pragma once

#include "LinearAlgebra.h"
#include <functional>

namespace NeuralNetwork {

class LossFunction {
public:
    using LossFunctionType = double(const Vector&, const Vector&);
    using GradienFunctionType = Matrix(const Vector&, const Vector&);

    enum Default {
        MSE,
        CrossEntropy
    };

    explicit LossFunction(LossFunctionType loss, GradienFunctionType gradient);

    explicit LossFunction(Default func);

    double Loss(const Vector& predicted, const Vector& target) const;

    Matrix Gradient(const Vector& predicted, const Vector& target) const;

    static double MSELoss(const Vector& predicted, const Vector& target);
    static Matrix MSEGradient(const Vector& predicted, const Vector& target);

    static double CrossEntropyLoss(const Vector& predicted, const Vector& target);
    static Matrix CrossEntropyGradient(const Vector& predicted, const Vector& target);
private:
    std::function<LossFunctionType> loss_;
    std::function<GradienFunctionType> gradient_;
};

}; // namespace NeuralNetwork