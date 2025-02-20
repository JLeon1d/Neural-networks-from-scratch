#include "neural_network.h"
#include "Eigen/Core"
#include "data_loader.h"
#include "loss.h"
#include <stdexcept>

#include <iostream>

namespace NeuralNetwork {

// maybe remove this constructor
Network::Network(size_t params_size, double learning_rate) : params_size_(params_size), learning_rate_(learning_rate) {
    net_.emplace_back(NeuralNetwork::LinearLayer(params_size, 7));
    net_.emplace_back(NeuralNetwork::NonLinearLayer(NeuralNetwork::NonLinearLayer::DefaultFunctions::ReLU));
    net_.emplace_back(NeuralNetwork::LinearLayer(7, 1));
}

Network::Network(const std::vector<size_t>& layer_sizes, double learning_rate) : learning_rate_(learning_rate) {
    if (layer_sizes.size() == 0) {
        throw std::logic_error("empty layer_sizes");
    }

    params_size_ = layer_sizes[0];
    for (size_t l_id = 0; l_id < layer_sizes.size() - 1; ++l_id) {
        net_.emplace_back(NeuralNetwork::LinearLayer(layer_sizes[l_id], layer_sizes[l_id + 1]));
        net_.emplace_back(NeuralNetwork::NonLinearLayer(NeuralNetwork::NonLinearLayer::DefaultFunctions::Sigmoid));
    }
    net_.pop_back();
    net_.emplace_back(NeuralNetwork::NonLinearLayer(NeuralNetwork::NonLinearLayer::DefaultFunctions::Softmax));
}

Network::Network(const std::vector<size_t>& layer_sizes,
                 const std::vector<NonLinearLayer::DefaultFunctions>& activation_functions,
                 double learning_rate) : learning_rate_(learning_rate) {
    if (layer_sizes.empty()) {
        throw std::logic_error("empty layer_sizes");
    } else if (activation_functions.empty()) {
        throw std::logic_error("empty activation_functions");
    } else if (layer_sizes.size() != activation_functions.size() + 1) {
        throw std::logic_error("layer_sizes and activation_function sizes mismatch");
    }

    params_size_ = layer_sizes[0];
    for (size_t l_id = 0; l_id < layer_sizes.size() - 1; ++l_id) {
        net_.emplace_back(NeuralNetwork::LinearLayer(layer_sizes[l_id], layer_sizes[l_id + 1]));
        net_.emplace_back(NeuralNetwork::NonLinearLayer(activation_functions[l_id]));
    }
}

double Network::TrainSingle(const DataSample& data_sample) {
    if (data_sample.first.size() != params_size_) {
        throw std::logic_error("wrong parametrs amount");
    }

    std::vector<Eigen::VectorXd> xs(net_.size() + 1);
    xs[0] = data_sample.first;

    for (size_t i = 0; i < net_.size(); ++i) {
        xs[i + 1] = net_[i]->Forward(xs[i]);
    }

    LossFunction loss_function(LossFunction::MSELoss, LossFunction::MSEGradient);

    auto loss = loss_function.Loss(xs[net_.size()], data_sample.second);
    auto u = loss_function.Gradient(xs[net_.size()], data_sample.second);
    for (int i = net_.size() - 1; i >= 0; --i) {
        u = net_[i]->Backward(xs[i], u, learning_rate_);
    }

    return loss;
}

double Network::TrainBatch(const std::vector<Eigen::VectorXd>& x, const std::vector<double>& y) {
    if (x.size() != params_size_) {
        throw std::logic_error("wrong parametrs amount");
    }
    if (x.size() != y.size()) {
        throw std::logic_error("x size not equal y size");
    }

    throw std::logic_error("not ready yet");
}

VectorXd Network::Predict(const VectorXd& features) {
    auto x = features;
    for (size_t i = 0; i < net_.size(); ++i) {
        x = net_[i]->Forward(x);
    }

    return x;
}

void Network::SetLearningRate(double learning_rate) {
    learning_rate_ = learning_rate;
}

};  // namespace NeuralNetwork