#pragma once

#include <Eigen/Dense>
#include <functional>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace NeuralNetwork {

class LinearLayer {
public:
    explicit LinearLayer(size_t in_size, size_t out_size);
    LinearLayer(LinearLayer&& oth);

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
        ReLU
    };

    explicit NonLinearLayer(DefaultFunctions f);

    template <typename F1, typename F2>
    explicit NonLinearLayer(F1 forward, F2 backward)
        : f_(std::move(forward)), b_(std::move(backward)) {
    }

    NonLinearLayer(NonLinearLayer&& oth);

    VectorXd Forward(const VectorXd& x) const;
    MatrixXd Backward(const VectorXd& x, const MatrixXd& u, long double lambda);

    // 1/(1 + e^(-x))

    static VectorXd Sigmoid(const VectorXd& x);
    static VectorXd SigmoidDeriv(const VectorXd& x);

    static VectorXd ReLU(const VectorXd& x);
    static VectorXd ReLuDeriv(const VectorXd& x);

private:
    std::function<VectorXd(VectorXd const&)> f_;
    std::function<MatrixXd(MatrixXd const&, MatrixXd const&, long double)> b_;
};

};  // namespace NeuralNetwork