#include "LinearAlgebra.h"
#include "neural_network.h"
#include "optimizer.h"
#include "loss.h"
#include "data_loader.h"
#include "mnist_data_loader.h"

#include <iostream>
#include <sys/types.h>
#include <chrono>

// take digit with max probability as predicted label
bool is_correct(const NeuralNetwork::Vector& predicted, const NeuralNetwork::Vector& target) {
    size_t answer = 0;
    for (size_t i = 1; i < predicted.size(); ++i) {
        if (predicted[i] > predicted[answer]) {
            answer = i;
        }
    }

    return (target[answer] == 1.0);
}

using ActivationType = NeuralNetwork::ActivationType;

// in /build: cmake .. && make && ./tests/MNIST/test_mnist
int main() {
    auto train_features_path = "../tests/MNIST/train-images.idx3-ubyte";
    auto train_lables_path = "../tests/MNIST/train-labels.idx1-ubyte";
    auto test_features_path = "../tests/MNIST/t10k-images.idx3-ubyte";
    auto test_lables_path = "../tests/MNIST/t10k-labels.idx1-ubyte";

    NeuralNetwork::DataLoader train_data_loader(train_features_path, train_lables_path, LoadMNIST);
    NeuralNetwork::DataLoader test_data_loader(test_features_path, test_lables_path, LoadMNIST);

    auto train_data = train_data_loader.Load();
    auto test_data = test_data_loader.Load();

    std::cout << "Loaded " << train_data.size() << " train data samples" << std::endl;
    std::cout << "Loaded " << test_data.size() << " test data samples" << std::endl;

    NeuralNetwork::Optimizer optimizer = NeuralNetwork::Optimizers::Adam(0.98, 0.98);

    NeuralNetwork::Network net({784, 300, 120, 10},
                               {ActivationType::Sigmoid, ActivationType::Sigmoid, ActivationType::Softmax}, 0.04,
                               NeuralNetwork::LossFunction(NeuralNetwork::LossType::Mse), std::move(optimizer));

    {  // calculate expected epoch time
        size_t sample_size = 100;
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < sample_size; ++i) {
            net.TrainSingle(train_data[i]);
        }
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        const double kMicrosecondsInMinute = 1e6 * 60.0;
        double expected_epoch_minutes =
            ((static_cast<double>(duration.count()) * train_data.size()) / static_cast<double>(sample_size)) /
            kMicrosecondsInMinute;
        std::cout << "Expected epoch time (minutes): " << expected_epoch_minutes << std::endl;
    }

    size_t progress_p = 1000;
    for (size_t epoch_id = 0; epoch_id < 30; ++epoch_id) {
        for (size_t i = 0; i < train_data.size(); ++i) {
            net.TrainSingle(train_data[i]);
            if (i % progress_p == 0) {
                std::cout << i << " / " << train_data.size() << std::endl;
            }
        }

        std::cout << "Epoch " << epoch_id + 1 << " is done" << std::endl;
        std::cout << "Accuracy train: " << net.CheckAccuracy(train_data, is_correct) << std::endl;
        std::cout << "Accuracy test:  " << net.CheckAccuracy(test_data, is_correct) << std::endl;
    }
}
