#include <iostream>
#include <sys/types.h>
#include <chrono>
#include "Eigen/Core"
#include "layer.h"
#include "loss.h"
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

    size_t TRAIN_SIZE = static_cast<double>(data.size()) * 0.8;
    NeuralNetwork::DataBatch train_data(TRAIN_SIZE);
    NeuralNetwork::DataBatch test_data(data.size() - TRAIN_SIZE);

    for (size_t i = 0; i < train_data.size(); ++i) {
        train_data[i] = std::move(data[i]);
    }
    for (size_t i = 0; i < test_data.size(); ++i) {
        test_data[i] = std::move(data[TRAIN_SIZE + i]);
    }

    NeuralNetwork::Network net(
        {784, 200, 80, 10},
        {ActivationFunctions::Sigmoid, ActivationFunctions::Sigmoid, ActivationFunctions::Softmax},
        0.05,
        NeuralNetwork::LossFunction(NeuralNetwork::LossFunction::Default::CrossEntropy)
    );

    {  // calculate expected epoch time
        size_t sample_size = 100;
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < sample_size; ++i) {
            net.TrainSingle(train_data[i]);
        }
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double expected_epoch_minutes = (((double)duration.count() * TRAIN_SIZE)/(double)sample_size) / 1e6 / 60.0;
        std::cout << "Expected epoch time (minutes): " << expected_epoch_minutes << std::endl;
    }

    size_t progress_p = 1000;
    size_t batch_size = 120;
    for (size_t epoch_id = 0; epoch_id < 30; ++epoch_id) {
        for (size_t i = 0; i < train_data.size(); ++i) {
            net.TrainSingle(train_data[i]);
            if (i % progress_p == 0) {
                std::cout << i << " / " << TRAIN_SIZE << std::endl;
            }
        }
        /*
        for (size_t i = 0; i + batch_size < train_data.size(); i += batch_size) {
            net.TrainBatch(std::vector<NeuralNetwork::DataSample>(train_data.begin() + i, train_data.begin() + i + batch_size));
            if (i % progress_p == 0) {
                std::cout << i << " / " << train_data.size() << std::endl;
            }
        }
        */

        std::cout << "Epoch " << epoch_id + 1 << " done" << std::endl;
        std::cout << "ACCURACY TRAIN: " << net.CheckAccuracy(train_data, is_correct) << std::endl;
        std::cout << "ACCURACY TEST:  " << net.CheckAccuracy(test_data, is_correct) << std::endl;
    }
}
