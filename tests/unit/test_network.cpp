#include <catch2/catch_test_macros.hpp>
#include "data_loader.h"
#include "neural_network.h"
#include "utils.h"
#include "loss.h"

TEST_CASE("Nertwork Basic", "[network]") {
    size_t in_size = 5;
    size_t mid_size = 3;
    size_t out_size = 2;
    NeuralNetwork::Optimizer optimizer = NeuralNetwork::Optimizers::Adam(0.98, 0.98);
    NeuralNetwork::Network net({in_size, mid_size, out_size},
                               {NeuralNetwork::ActivationType::Sigmoid, NeuralNetwork::ActivationType::Sigmoid}, 0.01,
                               NeuralNetwork::LossFunction(NeuralNetwork::LossType::Mse), std::move(optimizer));

    NeuralNetwork::DataBatch data = {
        {Utils::MakeVector({1, 2, 3, 4, 5}), Utils::MakeVector({3, 4})},
        {Utils::MakeVector({0.3, 4, 2, -0.3, 1}), Utils::MakeVector({0, -4})},
    };

    net.TrainSingle(data[0]);
    double loss0 = net.CheckLoss(data);
    net.SetLearningRate(0.1);

    auto predict = net.Predict(data[0].features);
    REQUIRE(predict.size() == 2);

    auto weights = net.GetWeights();
    REQUIRE(weights.size() == 2);
    REQUIRE((weights[0].A.cols() == 5 and weights[0].A.rows() == 3));
    REQUIRE(weights[0].b.size() == 3);

    REQUIRE((weights[1].A.cols() == 3 and weights[1].A.rows() == 2));
    REQUIRE(weights[1].b.size() == 2);
}
