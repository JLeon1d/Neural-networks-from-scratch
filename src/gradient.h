#pragma once

#include "LinearAlgebra.h"
#include "AnyMovable.h"

#include <initializer_list>
#include <variant>

namespace NeuralNetwork {

enum class OptimizerType { Classic, Momentum, AdaGrad, RMSProp, Adam };

struct Gradients {
    Matrix A_grad;
    Vector b_grad;
};

namespace Optimizers {

/// Cache here

namespace Caches {

struct Empty {};

struct Momentum {
    Matrix momentum_A_;
    Vector momentum_b_;
};

struct AdaGrad {
    Matrix A_g_;
    Vector b_g_;
};

struct RMSProp {
    Matrix A_g_;
    Vector b_g_;
};

struct Adam {
    Matrix A_linear_;
    Matrix A_square_;
    Vector b_linear_;
    Vector b_square_;
};

};  // namespace Caches

using Cache = std::variant<Caches::Empty, Caches::Momentum, Caches::AdaGrad, Caches::RMSProp, Caches::Adam>;

class Classic {
public:
    Gradients Optimize(const Vector& x, const RowVector& u, double learning_rate);
};

class Momentum {
public:
    static constexpr double kDefaultAlpha = 0.9;

    Momentum(size_t x_size, size_t u_size, double alpha = kDefaultAlpha);

    ~Momentum() = default;

    Gradients Optimize(const Vector& x, const RowVector& u, double learning_rate);

private:
    Matrix momentum_A_;
    Vector momentum_b_;

    double alpha_;
};

class AdaGrad {
public:
    static constexpr double epsilon = 1e-8;

    AdaGrad(size_t x_size, size_t u_size);

    ~AdaGrad() = default;

    Gradients Optimize(const Vector& x, const RowVector& u, double learning_rate);

private:
    Matrix A_g_;
    Vector b_g_;
};

class RMSProp {
public:
    static constexpr double kDefaultAlpha = 0.9;
    static constexpr double epsilon = 1e-8;

    RMSProp(size_t x_size, size_t u_size, double alpha = kDefaultAlpha);

    ~RMSProp() = default;

    Gradients Optimize(const Vector& x, const RowVector& u, double learning_rate);

private:
    Matrix A_g_;
    Vector b_g_;

    double alpha_;
};

// maybe make bias-correction (alpha ^ step) optional ?
class Adam {
public:
    static constexpr double kDefaultAlpha = 0.9;
    static constexpr double epsilon = 1e-8;

    Adam(size_t x_size, size_t u_size, double alpha_linear = kDefaultAlpha, double alpha_square = kDefaultAlpha);

    ~Adam() = default;

    Gradients Optimize(const Vector& x, const RowVector& u, double learning_rate);

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

};  // namespace Optimizers

template <class TBase>
class IGradientFunction : public TBase {
public:
    virtual Gradients Optimize(const Vector& x, const RowVector& u, double learning_rate) = 0;
};

template <class TBase, class TObject>
class CGradientFunctionImpl : public TBase {
    using CBase = TBase;

public:
    using CBase::CBase;

    Gradients Optimize(const Vector& x, const RowVector& u, double learning_rate) override {
        return CBase::Object().Optimize(x, u, learning_rate);
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
