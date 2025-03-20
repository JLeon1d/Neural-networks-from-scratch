#pragma once

#include "LinearAlgebra.h"
#include "AnyMovable.h"
#include "gradient.h"

#include <functional>

namespace NeuralNetwork {

class LinearLayer {
public:
    LinearLayer(size_t in_size, size_t out_size);

    LinearLayer(const LinearLayer& oth) = delete;
    LinearLayer(LinearLayer&& oth) = default;
    LinearLayer& operator=(const LinearLayer& oth) = delete;
    LinearLayer& operator=(LinearLayer&& oth) = default;

    ~LinearLayer() = default;

    Vector Forward(const Vector& x) const;
    Matrix Backward(const Vector& x, const RowVector& u, /* const */ GradientFunction& gf, double lambda);

private:
    size_t in_size_;
    size_t out_size_;
    Matrix A_;
    Vector b_;
};

class NonLinearLayer {
public:
    enum class DefaultFunctions {
        Sigmoid,
        ReLU,
        Softmax,
        LeakyReLU
    };

    explicit NonLinearLayer(DefaultFunctions f);

    template <typename F1, typename F2>
    NonLinearLayer(F1 forward, F2 backward)
        : f_(std::move(forward)), b_(std::move(backward)) {
    }

    NonLinearLayer(const NonLinearLayer& oth) = delete;
    NonLinearLayer(NonLinearLayer&& oth) noexcept = default;
    NonLinearLayer& operator=(const NonLinearLayer& oth) = delete;
    NonLinearLayer& operator=(NonLinearLayer&& oth) noexcept = default;

    Vector Forward(const Vector& x) const;
    Matrix Backward(const Vector& x, const RowVector& u, /* const */ GradientFunction& gf, double lambda);

    // maybe make those functions private?

    // 1/(1 + e^(-x))
    static Vector Sigmoid(const Vector& x);
    // returns derivative as matrix(1, x.size())
    static Matrix SigmoidDeriv(const Vector& x);

    // max(0, x)
    // Works very poorly (at least on mnist)
    static Vector ReLU(const Vector& x);
    static Matrix ReLUDeriv(const Vector& x);

    // sum x = 1.0
    static Vector Softmax(const Vector& x);
    static Matrix SoftmaxDeriv(const Vector& x);

    // maybe add parametr a : max(ax, x))
    // Works very poorly (at least on mnist)
    static Vector LeakyReLU(const Vector& x);
    static Matrix LeakyReLUDeriv(const Vector& x);  

private:
    std::function<Vector(Vector const&)> f_;
    std::function<Matrix(Matrix const&, Matrix const&)> b_;
};

namespace details {

template<class TBase>
class ILayer : public TBase {
public:
    virtual Vector Forward(const Vector& x) const = 0;
    virtual Matrix Backward(const Vector& x, const Matrix& u, /* const */ GradientFunction& gf, double lambda) = 0;
};

template<class TBase, class TObject>
class CLayerImpl : public TBase {
    using CBase = TBase;
public:
    using CBase::CBase;

    Vector Forward(const Vector& x) const override {
        return CBase::Object().Forward(x);
    }

    Matrix Backward(const Vector& x, const Matrix& u, /* const */ GradientFunction& gf, double lambda) override {
        return CBase::Object().Backward(x, u, gf, lambda);
    }
};

}; // namespace details

class Layer : public NSLibrary::CAnyMovable<details::ILayer, details::CLayerImpl> {
    using CBase = NSLibrary::CAnyMovable<details::ILayer, details::CLayerImpl>;
public:
    using CBase::CBase;
};

};  // namespace NeuralNetwork