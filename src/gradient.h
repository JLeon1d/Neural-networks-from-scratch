#pragma once

#include "LinearAlgebra.h"
#include "AnyMovable.h"

#include <initializer_list>

namespace NeuralNetwork {

using GradientPair = std::pair<Matrix, Vector>;
using FuncType = GradientPair(const Vector&, const Matrix&, double);

class ClassicGradient {
public:
    ClassicGradient() = default;

    ClassicGradient(const ClassicGradient& oth) = delete;
    ClassicGradient(ClassicGradient&& oth) = default;
    ClassicGradient& operator=(const ClassicGradient& oth) = delete;
    ClassicGradient& operator=(ClassicGradient&& oth) = default;

    GradientPair operator()(const Vector& x, const RowVector& u, double learning_rate);

private:
};

class MomentumGradient {
public:
    static constexpr double kDefaultAlpha = 0.9;

    MomentumGradient(size_t x_size, size_t u_size, double alpha = kDefaultAlpha);

    MomentumGradient(const MomentumGradient& oth) = delete;
    MomentumGradient(MomentumGradient&& oth) = default;
    MomentumGradient& operator=(const MomentumGradient& oth) = delete;
    MomentumGradient& operator=(MomentumGradient&& oth) = default;

    ~MomentumGradient() = default;

    GradientPair operator()(const Vector& x, const RowVector& u, double learning_rate);

private:
    Matrix momentum_A_;
    Vector momentum_b_;

    double alpha_;
};

class AdaGradient {
public:
    static constexpr double epsilon = 1e-8;

    AdaGradient(size_t x_size, size_t u_size);

    AdaGradient(const AdaGradient& oth) = delete;
    AdaGradient(AdaGradient&& oth) = default;
    AdaGradient& operator=(const AdaGradient& oth) = delete;
    AdaGradient& operator=(AdaGradient&& oth) = default;

    ~AdaGradient() = default;

    GradientPair operator()(const Vector& x, const RowVector& u, double learning_rate);

private:
    Matrix A_g_;
    Vector b_g_;
};

class RMSPropGradient {
public:
    static constexpr double kDefaultAlpha = 0.9;
    static constexpr double epsilon = 1e-8;

    RMSPropGradient(size_t x_size, size_t u_size, double alpha = kDefaultAlpha);

    RMSPropGradient(const RMSPropGradient& oth) = delete;
    RMSPropGradient(RMSPropGradient&& oth) = default;
    RMSPropGradient& operator=(const RMSPropGradient& oth) = delete;
    RMSPropGradient& operator=(RMSPropGradient&& oth) = default;
    ~RMSPropGradient() = default;

    GradientPair operator()(const Vector& x, const RowVector& u, double learning_rate);

private:
    Matrix A_g_;
    Vector b_g_;

    double alpha_;
};

// maybe make bias-correction (alpha ^ step) optional ?
class AdamGradient {
public:
    static constexpr double kDefaultAlpha = 0.9;
    static constexpr double epsilon = 1e-8;

    AdamGradient(size_t x_size, size_t u_size, double alpha_linear = kDefaultAlpha,
                 double alpha_square = kDefaultAlpha);

    AdamGradient(const AdamGradient& oth) = delete;
    AdamGradient(AdamGradient&& oth) = default;
    AdamGradient& operator=(const AdamGradient& oth) = delete;
    AdamGradient& operator=(AdamGradient&& oth) = default;

    ~AdamGradient() = default;

    GradientPair operator()(const Vector& x, const RowVector& u, double learning_rate);

private:
    Matrix A_linear_;
    Matrix A_square_;
    Vector b_linear_;
    Vector b_square_;

    double alpha_linear_;
    double alpha_square_;

    double alpha_linear_step_;  // alpha_linear_ ^ step
    double alpha_square_step_;  // alpha_square_ ^ step
};

template <class TBase>
class IGradientFunction : public TBase {
public:
    virtual GradientPair operator()(const Vector& x, const RowVector& u, double learning_rate) = 0;
};

template <class TBase, class TObject>
class CGradientFunctionImpl : public TBase {
    using CBase = TBase;

public:
    using CBase::CBase;

    GradientPair operator()(const Vector& x, const RowVector& u, double learning_rate) override {
        return CBase::Object()(x, u, learning_rate);
    }
};

class GradientFunction : public NSLibrary::CAnyMovable<IGradientFunction, CGradientFunctionImpl> {
    using CBase = NSLibrary::CAnyMovable<IGradientFunction, CGradientFunctionImpl>;

public:
    using CBase::CBase;

    enum Type { Classic, Momentum, AdaGrad, RMSProp, Adam };

    struct Initializer {
        Type type;
        std::initializer_list<double> args;
    };
};

};  // namespace NeuralNetwork
