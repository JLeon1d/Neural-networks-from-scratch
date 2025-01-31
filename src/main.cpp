#include <iostream>
#include "layer.h"
#include "loss.h"
#include "AnyMovable.h"

int main() {
    Eigen::MatrixXd m = Eigen::MatrixXd::Random(2, 3);
    std::cout << m << std::endl;
    m.transpose();
    std::cout << m << std::endl;
    std::cout << "WORKS\n";

    Eigen::VectorXd v = Eigen::VectorXd::Ones(3);
    v[1] += 1;
    v[2] += -2;
    std::cout << v << std::endl;
    /*
    std::cout << NeuralNetwork::NonLinearLayer::Sigmoid(v) << std::endl;
    std::cout << NeuralNetwork::NonLinearLayer::ReLU(v) << std::endl;
    NeuralNetwork::NonLinearLayer(NeuralNetwork::NonLinearLayer::DefaultFunctions::ReLU);
    */
    Eigen::VectorXd v2 = Eigen::VectorXd::Ones(3);
    std::cout << v2 << std::endl;
    std::cout << NeuralNetwork::LossFunction::Calculate(v, v2) << std::endl;
    
    return 0;
}