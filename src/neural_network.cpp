#include "neural_network.h"
#include "linear_layer.h"
#include "data_loader.h"
#include "gradient.h"
#include "loss.h"

namespace NeuralNetwork {

Network::Network(const std::vector<int64_t>& layer_sizes, const std::vector<ActivationType>& activation_functions,
                 double learning_rate, LossFunction lf, GradientFunction::Initializer gf_initializer)
    : learning_rate_(learning_rate), loss_function_(std::move(lf)) {
    assert(!layer_sizes.empty());
    assert(!activation_functions.empty());
    assert(layer_sizes.size() == activation_functions.size() + 1);

    for (size_t l_id = 0; l_id < layer_sizes.size() - 1; ++l_id) {
        net_.emplace_back(NeuralNetwork::LinearLayer(In{layer_sizes[l_id]}, Out{layer_sizes[l_id + 1]}));
        net_.emplace_back(NeuralNetwork::NonLinearLayer(activation_functions[l_id]));
        if (gf_initializer.type == GradientFunction::Type::Classic) {
            gradient_function_.emplace_back(ClassicGradient());
        } else if (gf_initializer.type == GradientFunction::Type::Momentum) {
            if (gf_initializer.args.size() > 0) {
                gradient_function_.emplace_back(
                    MomentumGradient(layer_sizes[l_id], layer_sizes[l_id + 1], *gf_initializer.args.begin()));
            } else {
                gradient_function_.emplace_back(MomentumGradient(layer_sizes[l_id], layer_sizes[l_id + 1]));
            }
        } else if (gf_initializer.type == GradientFunction::Type::RMSProp) {
            if (gf_initializer.args.size() > 0) {
                gradient_function_.emplace_back(
                    RMSPropGradient(layer_sizes[l_id], layer_sizes[l_id + 1], *gf_initializer.args.begin()));
            } else {
                gradient_function_.emplace_back(RMSPropGradient(layer_sizes[l_id], layer_sizes[l_id + 1]));
            }
        } else if (gf_initializer.type == GradientFunction::Type::Adam) {
            if (gf_initializer.args.size() > 1) {
                gradient_function_.emplace_back(AdamGradient(layer_sizes[l_id], layer_sizes[l_id + 1],
                                                             *gf_initializer.args.begin(),
                                                             *(gf_initializer.args.begin() + 1)));
            } else if (gf_initializer.args.size() > 0) {
                gradient_function_.emplace_back(
                    AdamGradient(layer_sizes[l_id], layer_sizes[l_id + 1], *gf_initializer.args.begin()));
            } else {
                gradient_function_.emplace_back(AdamGradient(layer_sizes[l_id], layer_sizes[l_id + 1]));
            }
        }

        // no weight updates for NonLinearLayer
        // kinda костыль
        gradient_function_.emplace_back(ClassicGradient());
    }
}

void Network::TrainSingle(const DataSample& data_sample) {
    std::vector<Vector> xs(net_.size() + 1);
    xs[0] = data_sample.features;

    for (size_t i = 0; i < net_.size(); ++i) {
        xs[i + 1] = net_[i]->Forward(xs[i]);
    }

    auto u = loss_function_.Gradient(xs[net_.size()], data_sample.target);
    for (int i = net_.size() - 1; i >= 0; --i) {
        u = net_[i]->Backward(xs[i], u, gradient_function_[i], learning_rate_);
    }
}

Vector Network::Predict(const Vector& features) const {
    auto x = features;
    for (size_t i = 0; i < net_.size(); ++i) {
        x = net_[i]->Forward(x);
    }

    return x;
}

double Network::CheckLoss(const std::vector<DataSample>& data) const {
    double loss_sum = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        auto predicted = Predict(data[i].features);
        loss_sum += loss_function_.Loss(predicted, data[i].target);
    }

    return loss_sum / static_cast<double>(data.size());
}

double Network::CheckAccuracy(const std::vector<DataSample>& data, std::function<Network::AccuracyPredicat> f) const {
    assert(f);

    size_t correct_count = 0;
    for (size_t i = 0; i < data.size(); ++i) {
        auto predicted = Predict(data[i].features);
        correct_count += f(predicted, data[i].target);
    }

    return static_cast<double>(correct_count) / static_cast<double>(data.size());
}

void Network::SetLearningRate(double learning_rate) {
    learning_rate_ = learning_rate;
}

};  // namespace NeuralNetwork
