#include <catch2/catch_test_macros.hpp>
#include <variant>
#include "LinearAlgebra.h"
#include "utils.h"
#include "optimizer.h"

class CustomOptimizer {
public:
    using RequiredCacheType = NeuralNetwork::Optimizers::Caches::SimpleState;

    static constexpr double kDefaultParam = 0.123;

    CustomOptimizer(double param = kDefaultParam) : param_(param) {
    }

    ~CustomOptimizer() = default;

    NeuralNetwork::Gradients Optimize(NeuralNetwork::Optimizers::Cache& cache, const NeuralNetwork::Vector& x,
                                      const NeuralNetwork::RowVector& u, double learning_rate) const {
        assert(std::holds_alternative<RequiredCacheType>(cache) && "Wrong type of cache");
        auto c_cache = std::get<RequiredCacheType>(cache);

        c_cache.A.setZero();
        c_cache.b.setConstant(4.0);
        return {u.transpose() * x.transpose(), param_ * c_cache.b};
    }

private:
    double param_;
};

TEST_CASE("Basic", "[optimizer]") {
    REQUIRE(std::is_same_v<NeuralNetwork::Optimizers::Classic::RequiredCacheType,
                           NeuralNetwork::Optimizers::Caches::Empty>);
    REQUIRE(std::is_same_v<NeuralNetwork::Optimizers::AdaGrad::RequiredCacheType,
                           NeuralNetwork::Optimizers::Caches::SimpleState>);
    REQUIRE(
        std::is_same_v<NeuralNetwork::Optimizers::Adam::RequiredCacheType, NeuralNetwork::Optimizers::Caches::Adam>);

    NeuralNetwork::Optimizer optimizer = NeuralNetwork::Optimizers::Classic();
    REQUIRE(optimizer.isDefined());
    optimizer.clear();
    REQUIRE(!optimizer.isDefined());

    optimizer = std::move(NeuralNetwork::Optimizers::RMSProp(0.98));
    REQUIRE(optimizer.isDefined());
    auto cache = optimizer->ConstructRequiredCache(2, 2);
    REQUIRE(std::holds_alternative<NeuralNetwork::Optimizers::RMSProp::RequiredCacheType>(cache));

    auto x = Utils::MakeVector({1, 2});
    auto u = Utils::MakeVector({3, 4});
    NeuralNetwork::Gradients grds = optimizer->Optimize(cache, x, u, 0.04);

    optimizer = std::move(NeuralNetwork::Optimizers::AdaGrad());
    grds = optimizer->Optimize(cache, x, u, 0.04);
}

TEST_CASE("Custom optimizer", "[optimizer]") {
    NeuralNetwork::Optimizer optimizer = std::move(CustomOptimizer(0.25));
    REQUIRE(optimizer.isDefined());

    auto cache = optimizer->ConstructRequiredCache(2, 2);
    REQUIRE(std::holds_alternative<NeuralNetwork::Optimizers::Caches::SimpleState>(cache));

    auto x = Utils::MakeVector({1, 2});
    auto u = Utils::MakeVector({3, 4});
    NeuralNetwork::Gradients grads = optimizer->Optimize(cache, x, u, 0.01);
    auto c_cache = std::get<NeuralNetwork::Optimizers::Caches::SimpleState>(cache);
    REQUIRE(c_cache.A == NeuralNetwork::Matrix::Zero(2, 2));
    REQUIRE(grads.b_grad == NeuralNetwork::Vector::Constant(2, 1.0));
}
