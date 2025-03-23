#pragma once

#include "LinearAlgebra.h"
#include "data_loader.h"
#include "layer.h"
#include "loss.h"
#include "gradient.h"

namespace NeuralNetwork {

// Cross Validation?
class Network {
public:
    static constexpr double kDefaultLearningRate = 0.2;

    Network(const std::vector<size_t>& layer_sizes, double learning_rate = kDefaultLearningRate,
            LossFunction lf = LossFunction(LossFunction::Default::MSE),
            GradientFunction::Type gf_type = GradientFunction::Type::Classic);

    // (layer_sizes.size()) should be equal to (activation_functions.size() + 1)
    Network(const std::vector<size_t>& layer_sizes,
            const std::vector<NonLinearLayer::DefaultFunctions>& activation_functions,
            double learning_rate = kDefaultLearningRate, LossFunction lf = LossFunction(LossFunction::Default::MSE),
            GradientFunction::Initializer gf_initializer = {GradientFunction::Type::Classic, {}});

    ~Network() = default;

    void TrainSingle(const DataSample& data_sample);

    Vector Predict(const Vector& features) const;

    // can paralel here, is it worth?
    double CheckLoss(const std::vector<DataSample>& data) const;

    using AccuracyPredicat = bool(const Vector&, const Vector&);
    double CheckAccuracy(const std::vector<DataSample>& data, std::function<AccuracyPredicat> f) const;

    void SetLearningRate(double learning_rate);

private:
    std::vector<Layer> net_;
    LossFunction loss_function_;
    std::vector<GradientFunction> gradient_function_;

    double learning_rate_ = 0.2;
};

};  // namespace NeuralNetwork
