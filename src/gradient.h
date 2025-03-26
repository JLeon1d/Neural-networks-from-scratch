#pragma once

#include "LinearAlgebra.h"
#include "AnyMovable.h"

#include <initializer_list>
#include <iostream>
#include <variant>

namespace NeuralNetwork {

enum class OptimizerType { Classic, Momentum, AdaGrad, RMSProp, Adam };

struct Gradients {
    Matrix A_grad;
    Vector b_grad;
};

namespace Optimizers {

namespace Caches {

struct Empty {
    Empty(size_t x_size, size_t u_size);
};

struct Momentum {
    Matrix momentum_A_;
    Vector momentum_b_;

    Momentum(size_t x_size, size_t u_size);
};

struct AdaGrad {
    Matrix A_g_;
    Vector b_g_;

    AdaGrad(size_t x_size, size_t u_size);
};

struct RMSProp {
    Matrix A_g_;
    Vector b_g_;

    RMSProp(size_t x_size, size_t u_size);
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

using Cache = std::variant<Caches::Empty, Caches::Momentum, Caches::AdaGrad, Caches::RMSProp, Caches::Adam>;

class Classic {
public:
    using RequiredCacheType = Caches::Empty;

    Gradients Optimize(Cache&, const Vector& x, const RowVector& u, double learning_rate) const;

    Cache ConstructCache(size_t x_size, size_t u_size) const {
        return RequiredCacheType(x_size, u_size);
    }

private:
};

class Momentum {
public:
    using RequiredCacheType = Caches::Momentum;

    static constexpr double kDefaultAlpha = 0.9;

    Momentum(size_t x_size, size_t u_size, double alpha = kDefaultAlpha);

    ~Momentum() = default;

    Gradients Optimize(Cache& cache, const Vector& x, const RowVector& u, double learning_rate) const;

private:
    double alpha_;
};

class AdaGrad {
public:
    using RequiredCacheType = Caches::AdaGrad;

    static constexpr double epsilon = 1e-8;

    AdaGrad(size_t x_size, size_t u_size);

    ~AdaGrad() = default;

    Gradients Optimize(Cache& cache, const Vector& x, const RowVector& u, double learning_rate) const;

private:
};

class RMSProp {
public:
    using RequiredCacheType = Caches::RMSProp;

    static constexpr double kDefaultAlpha = 0.9;
    static constexpr double epsilon = 1e-8;

    RMSProp(size_t x_size, size_t u_size, double alpha = kDefaultAlpha);

    ~RMSProp() = default;

    Gradients Optimize(Cache& cache, const Vector& x, const RowVector& u, double learning_rate) const;

private:
    double alpha_;
};

// maybe make bias-correction (alpha ^ step) optional ?
class Adam {
public:
    using RequiredCacheType = Caches::Adam;

    static constexpr double kDefaultAlpha = 0.9;
    static constexpr double epsilon = 1e-8;

    Adam(size_t x_size, size_t u_size, double alpha_linear = kDefaultAlpha, double alpha_square = kDefaultAlpha);

    ~Adam() = default;

    Gradients Optimize(Cache& cache, const Vector& x, const RowVector& u, double learning_rate) const;

    Cache ConstructCache(size_t x_size, size_t u_size) const {
        return RequiredCacheType(x_size, u_size);
    }

private:
    double alpha_linear_;
    double alpha_square_;
};

template <typename Opt>
struct CacheConstructor {
    using Type = Caches::Empty;

    Cache static Construct(size_t x_size, size_t u_size) {
        return Type(x_size, u_size);
    }
};

template <>
struct CacheConstructor<Momentum> {
    using Type = Caches::Momentum;

    Cache static Construct(size_t x_size, size_t u_size) {
        return Type(x_size, u_size);
    }
};

template <>
struct CacheConstructor<AdaGrad> {
    using Type = Caches::AdaGrad;

    Cache static Construct(size_t x_size, size_t u_size) {
        return Type(x_size, u_size);
    }
};

template <>
struct CacheConstructor<RMSProp> {
    using Type = Caches::RMSProp;

    Cache static Construct(size_t x_size, size_t u_size) {
        return Type(x_size, u_size);
    }
};

template <>
struct CacheConstructor<Adam> {
    using Type = Caches::Adam;

    Cache static Construct(size_t x_size, size_t u_size) {
        return Type(x_size, u_size);
    }
};

};  // namespace Optimizers

template <class TBase>
class IOptimizer : public TBase {
public:
    virtual Gradients Optimize(Optimizers::Cache& cache, const Vector& x, const RowVector& u,
                               double learning_rate) const = 0;

    virtual Optimizers::Cache ConstructCache(size_t x_size, size_t u_size) const = 0;
};

template <class TBase, class TObject>
class COptimizerImpl : public TBase {
    using CBase = TBase;

public:
    using CBase::CBase;
    // using RequiredCacheType = typename CBase::CObjectKeeper::CObjectType::RequiredCacheType;

    Gradients Optimize(Optimizers::Cache& cache, const Vector& x, const RowVector& u,
                       double learning_rate) const override {
        return CBase::Object().Optimize(cache, x, u, learning_rate);
    }

    Optimizers::Cache ConstructCache(size_t x_size, size_t u_size) const override {
        return CBase::Object().ConstructCache(x_size, u_size);
    }
};

class Optimizer : public NSLibrary::CAnyMovable<IOptimizer, COptimizerImpl> {
    using CBase = NSLibrary::CAnyMovable<IOptimizer, COptimizerImpl>;

public:
    using CBase::CBase;
};

};  // namespace NeuralNetwork
