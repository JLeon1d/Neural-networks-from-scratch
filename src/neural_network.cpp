#include "neural_network.h"
#include "Eigen/Core"
#include "data_loader.h"
#include "loss.h"
#include <stdexcept>

#include <iostream>

namespace NeuralNetwork {

Network::Network(const std::vector<size_t>& layer_sizes, double learning_rate, LossFunction lf)
    : learning_rate_(learning_rate), loss_function_(std::move(lf)) {
    if (layer_sizes.size() == 0) {
        throw std::logic_error("empty layer_sizes");
    }

    for (size_t l_id = 0; l_id < layer_sizes.size() - 1; ++l_id) {
        net_.emplace_back(NeuralNetwork::LinearLayer(layer_sizes[l_id], layer_sizes[l_id + 1]));
        net_.emplace_back(NeuralNetwork::NonLinearLayer(NeuralNetwork::NonLinearLayer::DefaultFunctions::Sigmoid));
    }
    net_.pop_back();
    net_.emplace_back(NeuralNetwork::NonLinearLayer(NeuralNetwork::NonLinearLayer::DefaultFunctions::Softmax));
}

Network::Network(const std::vector<size_t>& layer_sizes,
                 const std::vector<NonLinearLayer::DefaultFunctions>& activation_functions,
                 double learning_rate, LossFunction lf)
    : learning_rate_(learning_rate), loss_function_(std::move(lf)) {
    if (layer_sizes.empty()) {
        throw std::logic_error("empty layer_sizes");
    } else if (activation_functions.empty()) {
        throw std::logic_error("empty activation_functions");
    } else if (layer_sizes.size() != activation_functions.size() + 1) {
        throw std::logic_error("layer_sizes and activation_function sizes mismatch");
    }

    for (size_t l_id = 0; l_id < layer_sizes.size() - 1; ++l_id) {
        net_.emplace_back(NeuralNetwork::LinearLayer(layer_sizes[l_id], layer_sizes[l_id + 1]));
        net_.emplace_back(NeuralNetwork::NonLinearLayer(activation_functions[l_id]));
    }
}

void Network::TrainSingle(const DataSample& data_sample) {
    std::vector<Eigen::VectorXd> xs(net_.size() + 1);
    xs[0] = data_sample.features;

    for (size_t i = 0; i < net_.size(); ++i) {
        xs[i + 1] = net_[i]->Forward(xs[i]);
    }

    auto u = loss_function_.Gradient(xs[net_.size()], data_sample.target);
    for (int i = net_.size() - 1; i >= 0; --i) {
        u = net_[i]->Backward(xs[i], u, learning_rate_);
    }
}

void Network::TrainBatch(const DataBatch& data_batch) {
    if (data_batch.empty()) {
        throw std::logic_error("empty data batch");
    }

    std::vector<std::vector<Eigen::VectorXd>> xs(data_batch.size(), std::vector<Eigen::VectorXd>(net_.size() + 1));
    for (size_t sample_id = 0; sample_id < data_batch.size(); ++sample_id) {
        xs[sample_id][0] = data_batch[sample_id].features;
    }

    for (size_t sample_id = 0; sample_id < data_batch.size(); ++sample_id) {
        for (size_t i = 0; i < net_.size(); ++i) {
            xs[sample_id][i + 1] = net_[i]->Forward(xs[sample_id][i]);
        }
    }

    // initial gradient
    std::vector<MatrixXd> gradients(net_.size() + 1);
    gradients[net_.size()] = Eigen::MatrixXd::Zero(1, data_batch[0].target.size());
    for (size_t sample_id = 0; sample_id < data_batch.size(); ++sample_id) {
        gradients[net_.size()] += loss_function_.Gradient(xs[sample_id][net_.size()], data_batch[sample_id].target);
    }
    gradients[net_.size()] /= data_batch.size();

    // backpropagation
    for (int layer_id = net_.size() - 1; layer_id >= 0; --layer_id) {
        gradients[layer_id] = Eigen::MatrixXd::Zero(1, xs[0][layer_id].size());
        for (size_t sample_id = 0; sample_id < data_batch.size(); ++sample_id) {
            gradients[layer_id] += net_[layer_id]->Backward(xs[sample_id][layer_id], gradients[layer_id + 1], learning_rate_);
        }
        gradients[layer_id] /= data_batch.size();
    }
}

VectorXd Network::Predict(const VectorXd& features) const {
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