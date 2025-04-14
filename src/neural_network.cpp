#include "neural_network.h"
#include "linear_layer.h"

namespace NeuralNetwork {

Network::Network(const std::vector<size_t>& layer_sizes, const std::vector<ActivationType>& activation_functions,
                 double learning_rate, LossFunction lf, Optimizer&& optimizer)
    : learning_rate_(learning_rate), loss_function_(std::move(lf)), optimizer_(std::move(optimizer)) {
    assert(!layer_sizes.empty());
    assert(!activation_functions.empty());
    assert(layer_sizes.size() == activation_functions.size() + 1);

    for (size_t l_id = 0; l_id < layer_sizes.size() - 1; ++l_id) {
        net_.emplace_back(LinearLayer(In{layer_sizes[l_id]}, Out{layer_sizes[l_id + 1]}));
        optimizer_caches_.emplace_back(optimizer_->ConstructRequiredCache(layer_sizes[l_id], layer_sizes[l_id + 1]));

        net_.emplace_back(NonLinearLayer(activation_functions[l_id]));
        optimizer_caches_.emplace_back(Optimizers::Caches::Empty());
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
        u = net_[i]->Backward(xs[i], u, optimizer_, optimizer_caches_[i], learning_rate_);
    }
}

void Network::Train(const DataBatch& data) {
    for (const auto& data_sample : data) {
        TrainSingle(data_sample);
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

Network::NetworkWeights Network::GetWeights() const {
    NetworkWeights weights;
    for (size_t l_id = 0; l_id < net_.size(); l_id += 2) {
        weights.emplace_back(std::move(net_[l_id]->GetWeights()));
    }

    return weights;
}

};  // namespace NeuralNetwork
