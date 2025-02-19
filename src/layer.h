#pragma once

#include <Eigen/Dense>
#include <functional>
#include "AnyMovable.h"
#include "Eigen/Core"

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace NeuralNetwork {

class LinearLayer {
public:
    explicit LinearLayer(size_t in_size, size_t out_size);

    LinearLayer(const LinearLayer& oth) = delete;

    LinearLayer(LinearLayer&& oth) = default;

    LinearLayer& operator=(const LinearLayer& oth) = delete;

    LinearLayer& operator=(LinearLayer&& oth) = default;

    VectorXd Forward(const VectorXd& x) const;
    MatrixXd Backward(const VectorXd& x, const MatrixXd& u, long double lambda);

    ~LinearLayer() = default;

private:
    size_t in_size_;
    size_t out_size_;
    MatrixXd A;
    VectorXd b;
};

class NonLinearLayer {
public:
    enum class DefaultFunctions {
        Sigmoid,
        ReLU,
        Softmax
    };

    explicit NonLinearLayer(DefaultFunctions f);

    template <typename F1, typename F2>
    explicit NonLinearLayer(F1 forward, F2 backward)
        : f_(std::move(forward)), b_(std::move(backward)) {
    }

    NonLinearLayer(const NonLinearLayer& oth) = delete;

    NonLinearLayer(NonLinearLayer&& oth) noexcept = default;

    NonLinearLayer& operator=(const NonLinearLayer& oth) = delete;

    NonLinearLayer& operator=(NonLinearLayer&& oth) noexcept = default;

    VectorXd Forward(const VectorXd& x) const;
    MatrixXd Backward(const VectorXd& x, const MatrixXd& u, long double lambda);

    // 1/(1 + e^(-x))
    static VectorXd Sigmoid(const VectorXd& x);
    static MatrixXd SigmoidDeriv(const VectorXd& x);

    // max(0, x)
    // Works very poorly (at least on mnist)
    static VectorXd ReLU(const VectorXd& x);
    static MatrixXd ReLuDeriv(const VectorXd& x);

    // sum x = 1.0
    static VectorXd Softmax(const VectorXd& x);
    static MatrixXd SoftmaxDeriv(const VectorXd& x);

private:
    std::function<VectorXd(VectorXd const&)> f_;
    std::function<MatrixXd(MatrixXd const&, MatrixXd const&, long double)> b_;
};

template<class TBase>
class ILayer : public TBase {
public:
    virtual VectorXd Forward(const VectorXd& x) const = 0;
    virtual MatrixXd Backward(const VectorXd& x, const MatrixXd& u, long double lambda) = 0;
};

template<class TBase, class TObject>
class CLayerImpl : public TBase {
    using CBase = TBase;
public:
    using CBase::CBase;

    VectorXd Forward(const VectorXd& x) const override {
        return CBase::Object().Forward(x);
    }

    MatrixXd Backward(const VectorXd& x, const MatrixXd& u, long double lambda) override {
        return CBase::Object().Backward(x, u, lambda);
    }
};

class Layer : public NSLibrary::CAnyMovable<ILayer, CLayerImpl> {
    using CBase = NSLibrary::CAnyMovable<ILayer, CLayerImpl>;
public:
    using CBase::CBase;
};

};  // namespace NeuralNetwork