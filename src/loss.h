#pragma once

#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace NeuralNetwork {

class LossFunction {
public:
    static long double Calculate(const VectorXd& z, const VectorXd& y);

    static MatrixXd Gradient(const VectorXd& z, const VectorXd& y);
private:
};

}; // namespace NeuralNetwork