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
        {784, 200, 80, 10},
        {ActivationFunctions::Sigmoid, ActivationFunctions::Sigmoid, ActivationFunctions::Sigmoid},
        0.05
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
    for (size_t epoch_id = 0; epoch_id < 4; ++epoch_id) {
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
/*
notes for comparison:

LeakyReLU kinda slow

    net({784, 400, 80, 10}, {Sigmoid, Sigmoid, Softmax})
        l_rate = 0.1 :
            epoch 1: 0.58
            epoch 2: 0.71
            epoch 3: 0.764
        l_rate = 0.2 :
            epoch 1: 0.66 
            epoch 2: 0.745
            epoch 3: 0.75
        l_rate = 0.3 :
            epoch 1: 0.65

    net({784, 400, 80, 10}, {Sigmoid, Sigmoid, Sigmoid})
        l_rate = 0.01 :
            epoch 1: 0.17
            epoch 2: 0.22
            epoch 3: 0.28
        l_rate = 0.2 :
            epoch 1: 0.61
            epoch 2: 0.72
            epoch 3: 0.77

    [[trash]]
    net({784, 400, 80, 10}, {Softmax, Sigmoid, Softmax})
        l_rate = 0.2 :
            epoch 1: 0.12

    [[trash]]
    net({784, 400, 80, 10}, {LeakyReLU, LeakyReLU, Softmax})
        l_rate = 0.3 :
            epoch 1: 0.1

    net({784, 200, 80, 10}, {Sigmoid, Sigmoid, Sigmoid})
        l_rate = 0.05 :
            epoch 1: 0.36

*/