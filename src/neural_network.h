#pragma once

#include "Eigen/Core"
#include "data_loader.h"
#include "layer.h"
#include "loss.h"

using Eigen::VectorXd;

namespace NeuralNetwork {

// why do i have to reuse it?
using DataSample = NeuralNetwork::DataSample;

// Cross Validation
class Network {
public:
    static constexpr double kDefaultLearningRate = 0.2;

    explicit Network(const std::vector<size_t>& layer_sizes, double learning_rate=kDefaultLearningRate,
                     LossFunction lf=LossFunction(LossFunction::Default::MSE));

    // (layer_sizes.size()) should be equal to (activation_functions.size() + 1)
    explicit Network(const std::vector<size_t>& layer_sizes,
                     const std::vector<NonLinearLayer::DefaultFunctions>& activation_functions,
                     double learning_rate=kDefaultLearningRate,
                     LossFunction lf=LossFunction(LossFunction::Default::MSE));

    void TrainSingle(const DataSample& data_sample);

    void TrainBatch(const DataBatch& data_batch);

    VectorXd Predict(const VectorXd& features) const;
    
    // can paralel here, is it worth?
    double CheckLoss(const std::vector<DataSample>& data) const;

    using AccuracyPredicat = bool(const VectorXd&, const VectorXd&);
    double CheckAccuracy(const std::vector<DataSample>& data, std::function<AccuracyPredicat> f) const;

    void SetLearningRate(double learning_rate);

private:
    std::vector<Layer> net_;
    double learning_rate_ = 0.2;
    LossFunction loss_function_;
    // class for calculating Gradient stored here
};

}; // namespace NeuralNetwork