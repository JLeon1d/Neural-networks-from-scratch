#include <catch2/catch_test_macros.hpp>
#include "loss.h"

auto zero_loss = [](const NeuralNetwork::Vector& x, const NeuralNetwork::Vector& u) { return 0.0; };

auto zero_gradient = [](const NeuralNetwork::Vector& x, const NeuralNetwork::Vector& u) {
    NeuralNetwork::RowVector g = NeuralNetwork::RowVector::Zero(x.size());
    return g;
};

TEST_CASE("Simple", "[loss]") {
    auto lf = NeuralNetwork::LossFunction(NeuralNetwork::LossType::Mse);

    NeuralNetwork::Vector v1(3), v2(3);
    v1 << 1, 2, 3;
    v2 << -1, 0, 2;
    REQUIRE(lf.Loss(v1, v1) == 0.0);
    REQUIRE(lf.Loss(v1, v2) == 3.0);

    lf = NeuralNetwork::LossFunction(NeuralNetwork::LossType::CrossEntropy);
    REQUIRE(lf.Loss(v1, v1) == 0.0);  // how is it negative..
    REQUIRE(lf.Loss(v1, v2) == 3.0);

    auto lf_2 = NeuralNetwork::LossFunction::Mse();

    lf = NeuralNetwork::LossFunction(zero_loss, zero_gradient);
    lf = std::move(lf);
}

TEST_CASE("Custom methods", "[loss]") {
}
