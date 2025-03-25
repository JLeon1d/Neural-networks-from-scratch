#pragma once

#include "LinearAlgebra.h"
#include "gradient.h"

namespace NeuralNetwork {

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
