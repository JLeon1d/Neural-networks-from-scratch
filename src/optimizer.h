#pragma once

#include "LinearAlgebra.h"
#include "AnyMovable.h"

#include <variant>

namespace NeuralNetwork {

struct Gradients {
    Matrix A_grad;
    Vector b_grad;
};

namespace Optimizers {

namespace Caches {

struct Empty {
    Empty() = default;

    Empty(size_t x_size, size_t u_size);
};

struct SimpleState {
    Matrix A;
    Vector b;

    SimpleState(size_t x_size, size_t u_size);
};

struct Adam {
    Matrix A_linear_;
    Matrix A_square_;
    Vector b_linear_;
    Vector b_square_;

    double alpha_linear_step_;  // alpha_linear_ ^ step
    double alpha_square_step_;  // alpha_square_ ^ step

    Adam(size_t x_size, size_t u_size);
};

};  // namespace Caches

using Cache = std::variant<Caches::Empty, Caches::SimpleState, Caches::Adam>;

class Classic {
public:
    using RequiredCacheType = Caches::Empty;

    Gradients Optimize(Cache&, const Vector& x, const RowVector& u, double learning_rate) const;
};

class Momentum {
public:
    using RequiredCacheType = Caches::SimpleState;

    static constexpr double kDefaultAlpha = 0.9;

    Momentum(double alpha = kDefaultAlpha);

    ~Momentum() = default;

    Gradients Optimize(Cache& cache, const Vector& x, const RowVector& u, double learning_rate) const;

private:
    double alpha_;
};

class AdaGrad {
public:
    using RequiredCacheType = Caches::SimpleState;

    static constexpr double epsilon = 1e-8;

    Gradients Optimize(Cache& cache, const Vector& x, const RowVector& u, double learning_rate) const;
};

class RMSProp {
public:
    using RequiredCacheType = Caches::SimpleState;

    static constexpr double kDefaultAlpha = 0.9;
    static constexpr double epsilon = 1e-8;

    RMSProp(double alpha = kDefaultAlpha);

    ~RMSProp() = default;

    Gradients Optimize(Cache& cache, const Vector& x, const RowVector& u, double learning_rate) const;

private:
    double alpha_;
};

class Adam {
public:
    using RequiredCacheType = Caches::Adam;

    static constexpr double kDefaultAlpha = 0.9;
    static constexpr double epsilon = 1e-8;

    Adam(double alpha_linear = kDefaultAlpha, double alpha_square = kDefaultAlpha);

    ~Adam() = default;

    Gradients Optimize(Cache& cache, const Vector& x, const RowVector& u, double learning_rate) const;

private:
    double alpha_linear_;
    double alpha_square_;
};

};  // namespace Optimizers

namespace detail {

template <class TBase>
class IOptimizer : public TBase {
public:
    virtual Gradients Optimize(Optimizers::Cache& cache, const Vector& x, const RowVector& u,
                               double learning_rate) const = 0;

    virtual Optimizers::Cache ConstructRequiredCache(size_t x_size, size_t u_size) const = 0;
};

template <class TBase, class TObject>
class COptimizerImpl : public TBase {
    using CBase = TBase;

public:
    using CBase::CBase;
    using RequiredCacheType = typename CBase::CObjectKeeper::CObjectType::RequiredCacheType;

    Gradients Optimize(Optimizers::Cache& cache, const Vector& x, const RowVector& u,
                       double learning_rate) const override {
        return CBase::Object().Optimize(cache, x, u, learning_rate);
    }

    Optimizers::Cache ConstructRequiredCache(size_t x_size, size_t u_size) const override {
        return RequiredCacheType(x_size, u_size);
    }
};

};  // namespace detail

class Optimizer : public NSLibrary::CAnyMovable<detail::IOptimizer, detail::COptimizerImpl> {
    using CBase = NSLibrary::CAnyMovable<detail::IOptimizer, detail::COptimizerImpl>;

public:
    using CBase::CBase;
};

};  // namespace NeuralNetwork
