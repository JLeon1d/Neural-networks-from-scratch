#include <catch2/catch_test_macros.hpp>
#include "data_loader.h"
#include "neural_network.h"
#include "utils.h"
#include "loss.h"

TEST_CASE("Basic", "[network]") {
    NeuralNetwork::Optimizer optimizer = NeuralNetwork::Optimizers::Adam(0.98, 0.98);
    NeuralNetwork::Network net({5, 3, 2},
                               {NeuralNetwork::ActivationType::Sigmoid, NeuralNetwork::ActivationType::Sigmoid}, 0.01,
                               NeuralNetwork::LossFunction(NeuralNetwork::LossType::Mse), std::move(optimizer));

    NeuralNetwork::DataBatch data = {
        {Utils::MakeVector({1, 2, 3, 4, 5}), Utils::MakeVector({3, 4})},
        {Utils::MakeVector({0.3, 4, 2, -0.3, 1}), Utils::MakeVector({0, -4})},
    };

    double loss0 = net.CheckLoss(data);
    net.TrainSingle(data[0]);
}
