#pragma once

#include "Eigen/Core"
#include "layer.h"

namespace NeuralNetwork {

class Network {
public:
    explicit Network(size_t params_size);

    long double TrainSingle(const Eigen::VectorXd& x, long double y);

    long double TrainBatch(const std::vector<Eigen::VectorXd>& x, const std::vector<long double>& y);

    long double TestSingle(const Eigen::VectorXd& x, long double y);

    long double TestBatch(const std::vector<Eigen::VectorXd>& x, const std::vector<long double>& y);
private:
    size_t params_size_;
    std::vector<Layer> net_;
    // class for calculating Gradient stored here
};


}; // namespace NeuralNetwork