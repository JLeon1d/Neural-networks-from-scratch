#include "neural_network.h"
#include "Eigen/Core"
#include "loss.h"
#include <stdexcept>

namespace NeuralNetwork {

Network::Network(size_t params_size) : params_size_(params_size) {
    net_.emplace_back(NeuralNetwork::LinearLayer(params_size, 7));
    net_.emplace_back(NeuralNetwork::NonLinearLayer(NeuralNetwork::NonLinearLayer::DefaultFunctions::ReLU));
    net_.emplace_back(NeuralNetwork::LinearLayer(7, 1));
}

long double Network::TrainSingle(const Eigen::VectorXd& x, long double y) {
    if (x.size() != params_size_) {
        throw std::logic_error("wrong parametrs amount");
    }
        
    std::vector<Eigen::VectorXd> xs(net_.size() + 1);
    xs[0] = x;

    for (size_t i = 0; i < net_.size(); ++i) {
        xs[i + 1] = net_[i]->Forward(xs[i]);
    }

    auto loss = LossFunction::Calculate(xs[net_.size()], VectorXd::Constant(1, y));
    auto u = LossFunction::Gradient(xs[net_.size()], VectorXd::Constant(1, y));
    for (int i = net_.size() - 1; i >= 0; --i) {
        u = net_[i]->Backward(xs[i], u, 1.0);
    }

    return loss;
}

long double Network::TrainBatch(const std::vector<Eigen::VectorXd>& x, const std::vector<long double>& y) {
    if (x.size() != params_size_) {
        throw std::logic_error("wrong parametrs amount");
    }
    if (x.size() != y.size()) {
        throw std::logic_error("x size not equal y size");
    }

    throw std::logic_error("not ready yet");
}

long double Network::TestSingle(const Eigen::VectorXd& x, long double y) {
    std::vector<Eigen::VectorXd> xs(net_.size() + 1);
    xs[0] = x;

    for (size_t i = 0; i < net_.size(); ++i) {
        xs[i + 1] = net_[i]->Forward(xs[i]);
    }

    return LossFunction::Calculate(xs[net_.size()], VectorXd::Constant(1, y));
}

long double Network::TestBatch(const std::vector<Eigen::VectorXd>& x, const std::vector<long double>& y) {
    if (x.size() != params_size_) {
        throw std::logic_error("wrong parametrs amount");
    }
    if (x.size() != y.size()) {
        throw std::logic_error("x size not equal y size");
    }

    return 0.0;
}

};  // namespace NeuralNetwork