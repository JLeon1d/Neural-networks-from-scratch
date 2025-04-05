#include <catch2/catch_test_macros.hpp>
#include "LinearAlgebra.h"
#include "linear_layer.h"
#include "non_linear_layer.h"
#include "optimizer.h"
#include "utils.h"
#include "layer.h"

TEST_CASE("LinearLayer Basic", "[layer]") {
    size_t in_size = 5, out_size = 2;
    NeuralNetwork::Layer l = NeuralNetwork::LinearLayer(NeuralNetwork::In{in_size}, NeuralNetwork::Out{out_size});
    REQUIRE(l.isDefined());

    auto x = Utils::MakeVector({1, 2, 3, 4, 5});
    auto y = l->Forward(x);
    REQUIRE(y.size() == out_size);

    NeuralNetwork::Optimizer optimizer = NeuralNetwork::Optimizers::Classic();
    auto cache = optimizer->ConstructRequiredCache(in_size, out_size);
    auto u = l->Backward(x, Utils::MakeRowVector({1, 2}), optimizer, cache, 0.1);
    REQUIRE(u.size() == in_size);

    auto weights = l->GetWeights();
    REQUIRE((weights.A.cols() == in_size and weights.A.rows() == out_size));
    REQUIRE(weights.b.size() == out_size);
}

TEST_CASE("NonLinearLayer Basic", "[layer]") {
    NeuralNetwork::Layer l = NeuralNetwork::NonLinearLayer(NeuralNetwork::ActivationType::Sigmoid);
    REQUIRE(l.isDefined());

    l = std::move(NeuralNetwork::NonLinearLayer::Softmax());
    REQUIRE(l.isDefined());
    l = NeuralNetwork::NonLinearLayer::LeakyReLU(0.4);
    REQUIRE(l.isDefined());

    auto x = Utils::MakeVector({1, 2, 3, 4, 5});
    auto y = l->Forward(x);
    REQUIRE(y.size() == x.size());

    NeuralNetwork::Optimizer optimizer = NeuralNetwork::Optimizers::Classic();
    auto cache = optimizer->ConstructRequiredCache(2, 4);
    auto u = l->Backward(x, Utils::MakeRowVector({1, 2, 3, 2, 1}), optimizer, cache, 0.1);
    REQUIRE(u.cols() == 5);

    auto weights = l->GetWeights();
    REQUIRE((weights.A.cols() == 0 and weights.A.rows() == 0));
    REQUIRE(weights.b.size() == 0);
}

TEST_CASE("Layer custom activation function", "[layer]") {
    auto zero_forward = [](const NeuralNetwork::Vector& x) -> NeuralNetwork::Vector {
        return NeuralNetwork::Vector::Zero(x.size());
    };

    auto zero_backward = [](const NeuralNetwork::Vector& x,
                            const NeuralNetwork::RowVector& u) -> NeuralNetwork::RowVector {
        return NeuralNetwork::RowVector::Zero(x.size());
    };

    NeuralNetwork::Layer l = NeuralNetwork::NonLinearLayer(zero_forward, zero_backward);
    REQUIRE(l.isDefined());

    auto x = Utils::MakeVector({1, 2, 3, 4, 5});
    auto y = l->Forward(x);
    REQUIRE(y.size() == x.size());
    REQUIRE(y == NeuralNetwork::Vector::Zero(y.size()));

    NeuralNetwork::Optimizer optimizer = NeuralNetwork::Optimizers::Classic();
    auto cache = optimizer->ConstructRequiredCache(2, 4);
    auto u = l->Backward(x, Utils::MakeRowVector({1, 2, 3, 2, 1}), optimizer, cache, 0.1);
    REQUIRE(u.cols() == 5);
    REQUIRE(u == NeuralNetwork::RowVector::Zero(u.size()));
}
