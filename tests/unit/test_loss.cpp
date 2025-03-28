#include <catch2/catch_test_macros.hpp>
#include "utils.h"
#include "loss.h"

TEST_CASE("Basic", "[loss]") {
    auto lf = NeuralNetwork::LossFunction(NeuralNetwork::LossType::Mse);

    auto v1 = Utils::MakeVector({1, 2, 3});
    auto v2 = Utils::MakeVector({-1, 0, 2});
    REQUIRE(lf.Loss(v1, v1) == 0.0);
    REQUIRE(lf.Loss(v1, v2) == 3.0);

    auto v3 = Utils::MakeRowVector({4.0 / 3.0, 4.0 / 3.0, 2.0 / 3.0});
    REQUIRE(lf.Gradient(v1, v2) == v3);

    lf = NeuralNetwork::LossFunction(NeuralNetwork::LossType::CrossEntropy);
    REQUIRE(lf.Loss(v1, v2) == -(log(v1[0]) * v2[0] + log(v1[1]) * v2[1] + log(v1[2]) * v2[2]));

    auto lf_2 = NeuralNetwork::LossFunction::CrossEntropy();
    REQUIRE(lf.Loss(v1, v2) == lf_2.Loss(v1, v2));
}

TEST_CASE("Custom functions", "[loss]") {
    auto zero_loss = [](const NeuralNetwork::Vector& x, const NeuralNetwork::Vector& u) { return 0.0; };

    auto zero_gradient = [](const NeuralNetwork::Vector& x, const NeuralNetwork::Vector& u) {
        NeuralNetwork::RowVector g = NeuralNetwork::RowVector::Zero(x.size());
        return g;
    };

    auto lf = NeuralNetwork::LossFunction(zero_loss, zero_gradient);
    NeuralNetwork::Vector v1 = NeuralNetwork::Vector::Random(100);
    NeuralNetwork::Vector v2 = NeuralNetwork::Vector::Random(100);
    REQUIRE(lf.Loss(v1, v2) == 0.0);
    REQUIRE(lf.Gradient(v1, v2) == NeuralNetwork::RowVector::Zero(100));
}
