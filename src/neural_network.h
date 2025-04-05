#pragma once

#include <vector>
#include "LinearAlgebra.h"
#include "data_loader.h"
#include "non_linear_layer.h"
#include "loss.h"
#include "optimizer.h"

namespace NeuralNetwork {

class Network {
public:
    static constexpr double kDefaultLearningRate = 0.2;

    // (layer_sizes.size()) should be equal to (activation_functions.size() + 1)
    Network(const std::vector<size_t>& layer_sizes, const std::vector<ActivationType>& activation_functions,
            double learning_rate = kDefaultLearningRate, LossFunction lf = LossFunction(LossType::Mse),
            Optimizer&& optimizer = std::move(Optimizers::Classic()));

    ~Network() = default;

    void TrainSingle(const DataSample& data_sample);

    Vector Predict(const Vector& features) const;

    double CheckLoss(const std::vector<DataSample>& data) const;

    using AccuracyPredicat = bool(const Vector&, const Vector&);
    double CheckAccuracy(const std::vector<DataSample>& data, std::function<AccuracyPredicat> f) const;

    void SetLearningRate(double learning_rate);

    using NetworkWeights = std::vector<LayerWeights>;
    NetworkWeights GetWeights() const;

private:
    std::vector<Layer> net_;
    LossFunction loss_function_;
    Optimizer optimizer_;
    std::vector<Optimizers::Cache> optimizer_caches_;

    double learning_rate_ = 0.2;
};

};  // namespace NeuralNetwork
