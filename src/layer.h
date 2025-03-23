#pragma once

#include "LinearAlgebra.h"
#include "AnyMovable.h"
#include "gradient.h"

#include <functional>

namespace NeuralNetwork {

enum class ActivationType { Sigmoid, ReLU, Softmax, LeakyReLU };

template <ActivationType type>
struct ActivationData;

template <>
struct ActivationData<ActivationType::Sigmoid> {
    static Vector forward(const Vector& x) {
        return (1.0 / (1.0 + (-x.array()).exp())).matrix();
    }

    static RowVector backward(const Vector& x, const RowVector& u) {
        assert(u.size() == x.size());
        Matrix deriv = Matrix(forward(x).array() * (-forward(x).array() + 1)).transpose();
        return u.cwiseProduct(deriv);
    }
};

template <>
struct ActivationData<ActivationType::ReLU> {
    static Vector forward(const Vector& x) {
        return (x.array().max(0)).matrix();
    }

    static RowVector backward(const Vector& x, const RowVector& u) {
        assert(u.size() == x.size());
        Matrix deriv = (0.5 + (x.array().sign()) / 2.0).matrix().asDiagonal();
        return u * deriv;
    }
};

template <>
struct ActivationData<ActivationType::Softmax> {
    static Vector forward(const Vector& x) {
        double m = x.maxCoeff();  // to prevent overflow
        Vector exp_x = (x.array() - m).exp();
        double exp_sum = exp_x.sum();
        return exp_x / exp_sum;
    }

    static RowVector backward(const Vector& x, const RowVector& u) {
        assert(u.size() == x.size());

        Vector s = forward(x);
        Matrix deriv = -s * s.transpose();  // -s_i * s_j
        deriv.diagonal() += s;              // -s_i * s_j + (i == j) * s_i

        return u * deriv;
    }
};

struct ActivationSelector {
    using ForwardType = Vector(Vector const&);
    using BackwardType = RowVector(Vector const&, RowVector const&);

    static ForwardType* forward(ActivationType type) {
        ForwardType* ptr;
        switch (type) {
            case ActivationType::Sigmoid:
                ptr = &ActivationData<ActivationType::Sigmoid>::forward;
                break;
            case ActivationType::ReLU:
                ptr = &ActivationData<ActivationType::ReLU>::forward;
                break;
            case ActivationType::Softmax:
                ptr = &ActivationData<ActivationType::Softmax>::forward;
                break;
            default:
                assert(false && "Usupported ActivationType");
                break;
        }

        return ptr;
    }

    static BackwardType* backward(ActivationType type) {
        BackwardType* ptr;
        switch (type) {
            case ActivationType::Sigmoid:
                ptr = &ActivationData<ActivationType::Sigmoid>::backward;
                break;
            case ActivationType::ReLU:
                ptr = &ActivationData<ActivationType::ReLU>::backward;
                break;
            case ActivationType::Softmax:
                ptr = &ActivationData<ActivationType::Softmax>::backward;
                break;
            default:
                assert(false && "Usupported ActivationType");
                break;
        }

        return ptr;
    }
};

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
    Matrix A_;
    Vector b_;
};

class NonLinearLayer {
public:
    using ForwardType = ActivationSelector::ForwardType;
    using BackwardType = ActivationSelector::BackwardType;

    explicit NonLinearLayer(ActivationType type);

    NonLinearLayer(ForwardType forward, BackwardType backward);

    NonLinearLayer(const NonLinearLayer& oth) = delete;
    NonLinearLayer(NonLinearLayer&& oth) noexcept = default;
    NonLinearLayer& operator=(const NonLinearLayer& oth) = delete;
    NonLinearLayer& operator=(NonLinearLayer&& oth) noexcept = default;

    Vector Forward(const Vector& x) const;
    Matrix Backward(const Vector& x, const RowVector& u, /* const */ GradientFunction& gf, double lambda);

private:
    std::function<ForwardType> forward_;
    std::function<BackwardType> backward_;
};

namespace details {

template <class TBase>
class ILayer : public TBase {
public:
    virtual Vector Forward(const Vector& x) const = 0;
    virtual Matrix Backward(const Vector& x, const Matrix& u, /* const */ GradientFunction& gf, double lambda) = 0;
};

template <class TBase, class TObject>
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

};  // namespace details

class Layer : public NSLibrary::CAnyMovable<details::ILayer, details::CLayerImpl> {
    using CBase = NSLibrary::CAnyMovable<details::ILayer, details::CLayerImpl>;

public:
    using CBase::CBase;
};

};  // namespace NeuralNetwork
