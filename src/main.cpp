#include <iostream>
#include <Eigen/Dense>
#include <stdexcept>
#include "layer.h"
#include "loss.h"

long double Learn(std::vector<NeuralNetwork::Layer>& NetWork, const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
    if (x.size() != y.size()) {
        throw std::logic_error("Wrong input sizes");
    }

    size_t batch_size = x.size();
    std::vector<Eigen::VectorXd> xs(NetWork.size() + 1);
    xs[0] = x;

    for (size_t i = 0; i < NetWork.size(); ++i) {
        xs[i + 1] = NetWork[i]->Forward(xs[i]);
    }

    auto loss = NeuralNetwork::LossFunction::Calculate(xs[NetWork.size()], y);
    auto u = NeuralNetwork::LossFunction::Gradient(xs[NetWork.size()], y);
    for (int i = NetWork.size() - 1; i >= 0; --i) {
        u = NetWork[i]->Backward(xs[i], u, 1.0);
    }

    return loss;
}

int main() {  // for now just an example of work
    size_t batch_size = 5;

    std::vector<NeuralNetwork::Layer> NetWork;
    NetWork.emplace_back(NeuralNetwork::LinearLayer(batch_size, 7));
    NetWork.emplace_back(NeuralNetwork::NonLinearLayer(NeuralNetwork::NonLinearLayer::DefaultFunctions::ReLU));
    NetWork.emplace_back(NeuralNetwork::LinearLayer(7, batch_size));
    NetWork.emplace_back(NeuralNetwork::NonLinearLayer(NeuralNetwork::NonLinearLayer::DefaultFunctions::Sigmoid));

    Eigen::VectorXd x = Eigen::VectorXd::Ones(batch_size);
    Eigen::VectorXd y(Eigen::VectorXd::Random(batch_size));

    auto loss = Learn(NetWork, x, y);

    std::cout << "WORKS, loss: " << loss << std::endl;
    return 0;
}