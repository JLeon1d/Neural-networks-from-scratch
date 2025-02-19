#pragma once

#include <Eigen/Dense>
#include <functional>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace NeuralNetwork {

class LossFunction {
public:
    using LossFunctionType = double(const VectorXd&, const VectorXd&);
    using GradienFunctionType = MatrixXd(const VectorXd&, const VectorXd&);

    explicit LossFunction(LossFunctionType loss, GradienFunctionType gradient);

    double Loss(const VectorXd& predicted, const VectorXd& target);

    MatrixXd Gradient(const VectorXd& predicted, const VectorXd& target);

    static double MSELoss(const VectorXd& predicted, const VectorXd& target);

    static MatrixXd MSEGradient(const VectorXd& predicted, const VectorXd& target);
private:
    std::function<LossFunctionType> loss_;
    std::function<GradienFunctionType> gradient_;
};

}; // namespace NeuralNetwork