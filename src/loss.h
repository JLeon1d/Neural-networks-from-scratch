#pragma once

#include "LinearAlgebra.h"
#include <functional>

namespace NeuralNetwork {

class LossFunction {
public:
    using LossFunctionType = double(const Vector&, const Vector&);
    using GradienFunctionType = RowVector(const Vector&, const Vector&);

    enum Default { MSE, CrossEntropy };

    explicit LossFunction(Default func);
    LossFunction(LossFunctionType loss, GradienFunctionType gradient);

    ~LossFunction() = default;

    double Loss(const Vector& predicted, const Vector& target) const;

    RowVector Gradient(const Vector& predicted, const Vector& target) const;

    static double MSELoss(const Vector& predicted, const Vector& target);
    static RowVector MSEGradient(const Vector& predicted, const Vector& target);

    static double CrossEntropyLoss(const Vector& predicted, const Vector& target);
    static RowVector CrossEntropyGradient(const Vector& predicted, const Vector& target);

private:
    std::function<LossFunctionType> loss_;
    std::function<GradienFunctionType> gradient_;
};

};  // namespace NeuralNetwork
