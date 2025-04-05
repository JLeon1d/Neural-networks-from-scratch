#include <catch2/catch_test_macros.hpp>
#include "data_loader.h"

TEST_CASE("Basic", "[data_loader]") {
    size_t data_size = 4;
    size_t features_size = 5;
    size_t target_size = 2;
    // example function which fills data with random numbers
    auto loader_function = [data_size, features_size, target_size](const std::string&, const std::string) {
        std::vector<std::pair<std::vector<double>, std::vector<double>>> data(data_size);
        for (size_t i = 0; i < data_size; ++i) {
            std::vector<double> features(features_size);
            std::vector<double> target(target_size);

            for (auto& x : features) {
                x = rand();
            }
            for (auto& x : target) {
                x = rand();
            }

            data[i] = {features, target};
        }

        return data;
    };

    NeuralNetwork::DataLoader data_loader("some/path", "some/other/path", loader_function);

    NeuralNetwork::DataBatch data = data_loader.Load();
    REQUIRE(data.size() == data_size);
    REQUIRE(data[0].features.size() == features_size);
    REQUIRE(data[0].target.size() == target_size);
}
