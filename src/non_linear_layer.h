#pragma once

#include "LinearAlgebra.h"
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
        auto exp_x = (x.array() - m).exp();
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

class NonLinearLayer {
public:
    using ForwardType = ActivationSelector::ForwardType;
    using BackwardType = ActivationSelector::BackwardType;

    explicit NonLinearLayer(ActivationType type);
    NonLinearLayer(ForwardType forward, BackwardType backward);

    Vector Forward(const Vector& x) const;
    Matrix Backward(const Vector& x, const RowVector& u, /* const */ GradientFunction& gf, double lambda);

private:
    std::function<ForwardType> forward_;
    std::function<BackwardType> backward_;
};

};  // namespace NeuralNetwork
