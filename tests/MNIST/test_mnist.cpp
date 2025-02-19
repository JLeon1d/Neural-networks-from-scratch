#include <iostream>
#include <sys/types.h>
#include <chrono>
#include "Eigen/Core"
#include "layer.h"
#include "neural_network.h"
#include "data_loader.h"
#include "mnist_data_loader.h"

bool is_correct(const VectorXd& predicted, const VectorXd& target) {
    size_t answer = 0;
    for (size_t i = 1; i < predicted.size(); ++i) {
        if (predicted[i] > predicted[answer]) {
            answer = i;
        }
    }

    return (target[answer] == 1.0);
}

using ActivationFunctions = NeuralNetwork::NonLinearLayer::DefaultFunctions;


// in /build: cmake .. && make && ./test_mnist
int main() {
    auto features_path = "../tests/MNIST/t10k-images.idx3-ubyte";
    auto lables_path = "../tests/MNIST/t10k-labels.idx1-ubyte";
    NeuralNetwork::DataLoader data_loader(features_path, lables_path, LoadMNIST);

    auto data = data_loader.Load();
    std::cout << "Loaded " << data.size() << " data samples" << std::endl;

    NeuralNetwork::Network net(
        {784, 400, 50, 10},
        {ActivationFunctions::Sigmoid, ActivationFunctions::Sigmoid, ActivationFunctions::Softmax}    
    );
    size_t TRAIN_SIZE = ((double)data.size()) * 0.8;

    {  // calculate expected epoch time
        size_t sample_size = 100;
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < sample_size; ++i) {
            net.TrainSingle(data[i]);
        }
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double expected_epoch_minutes = (((double)duration.count() * TRAIN_SIZE)/(double)sample_size) / 1e6 / 60.0;
        std::cout << "Expected epoch time (minutes): " << expected_epoch_minutes << std::endl;
    }

    size_t progress_p = 1000;
    for (size_t epoch_id = 0; epoch_id < 3; ++epoch_id) {
        std::cout << "Epoch " << epoch_id + 1 << std::endl;
        for (size_t i = 0; i < TRAIN_SIZE; ++i) {
            net.TrainSingle(data[i]);
            if (i % progress_p == 0) {
                std::cout << i << " / " << TRAIN_SIZE << std::endl;
            }
        }
        std::cout << "Done Training" << std::endl;

        size_t cnt = 0;
        size_t correct_count = 0;
        for (size_t i = TRAIN_SIZE; i < data.size(); ++i) {
            auto predicted = net.Predict(data[i].first);
            correct_count += is_correct(predicted, data[i].second);
        }

        std::cout << "ACCURACY: " << (double)correct_count/(double)(data.size() - TRAIN_SIZE) << std::endl;
    }
}