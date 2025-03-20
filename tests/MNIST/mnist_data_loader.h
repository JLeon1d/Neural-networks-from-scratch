#pragma once

#include <cstdint>
#include <string>
#include <vector>

// not to confuse with NeuralNetwork::DataSample
using DataSample = std::pair<std::vector<double>, std::vector<double>>; 

uint32_t reverse_int(uint32_t i);

std::vector<std::vector<uint8_t>> read_mnist_images(const std::string& path);

std::vector<uint8_t> read_mnist_labels(const std::string& path);

std::vector<DataSample> LoadMNIST(const std::string& features_path, const std::string& lables_path);
