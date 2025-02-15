#include <iostream>
#include <Eigen/Dense>
#include "neural_network.h"

int main() {  // for now just an example of work
    size_t params_size = 5;
    NeuralNetwork::Network net(params_size);

    Eigen::VectorXd x = Eigen::VectorXd::Ones(params_size);
    long double y = 0.3;

    auto train_loss = net.TrainSingle(x, y);
    std::cout << "TRAIN WORKS, loss: " << train_loss << std::endl;

    x = Eigen::VectorXd::Ones(params_size);
    auto test_loss = net.TestSingle(x, 0.5);
    std::cout << "TEST WORKS, loss: " << test_loss << std::endl;

    return 0;
}