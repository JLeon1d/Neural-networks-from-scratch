#pragma once

#include "Eigen/Core"
#include "layer.h"

using Eigen::VectorXd;

namespace NeuralNetwork {

// why do i have to reuse it?
// Добавить структуру вместо пары
using DataSample = std::pair<VectorXd, VectorXd>;

// need to have option to add custom loss function
class Network {
public:
    explicit Network(size_t params_size);

    explicit Network(const std::vector<size_t>& layer_sizes);

    // (layer_sizes.size()) should be equal to (activation_functions.size() + 1)
    explicit Network(const std::vector<size_t>& layer_sizes,
                     const std::vector<NonLinearLayer::DefaultFunctions>& activation_functions);

    double TrainSingle(const DataSample& data_sample);

    double TrainBatch(const std::vector<VectorXd>& x, const std::vector<double>& y);

    VectorXd Predict(const VectorXd& features);

private:
    size_t params_size_;
    std::vector<Layer> net_;
    // class for calculating Gradient stored here
};


}; // namespace NeuralNetwork