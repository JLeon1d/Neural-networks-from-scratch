#include "mnist_data_loader.h"
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <iostream>

uint32_t reverse_int(uint32_t i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((uint32_t)c1 << 24) + ((uint32_t)c2 << 16) + ((uint32_t)c3 << 8) + c4;
}

// Function to read MNIST images
std::vector<std::vector<uint8_t>> read_mnist_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        if (file.fail()) {
            std::cout << "Error details: " << strerror(errno) << std::endl;
            std::cout << std::filesystem::current_path() << std::endl;
        }

        throw std::runtime_error("Could not open file: " + path);
    }

    uint32_t magic_number, num_images, num_rows, num_cols;
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&num_images, sizeof(num_images));
    file.read((char*)&num_rows, sizeof(num_rows));
    file.read((char*)&num_cols, sizeof(num_cols));

    magic_number = reverse_int(magic_number);
    num_images = reverse_int(num_images);
    num_rows = reverse_int(num_rows);
    num_cols = reverse_int(num_cols);

    std::vector<std::vector<uint8_t>> images(num_images, std::vector<uint8_t>(num_rows * num_cols));
    for (uint32_t i = 0; i < num_images; ++i) {
        file.read((char*)images[i].data(), num_rows * num_cols);
    }

    return images;
}

// Function to read MNIST labels
std::vector<uint8_t> read_mnist_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open file: " + path);
    }

    uint32_t magic_number, num_labels;
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&num_labels, sizeof(num_labels));

    magic_number = reverse_int(magic_number);
    num_labels = reverse_int(num_labels);

    std::vector<uint8_t> labels(num_labels);
    file.read((char*)labels.data(), num_labels);

    return labels;
}

// data[i].second - vector with 1 in index of right value, 0 everywhere else
std::vector<DataSample> LoadMNIST(const std::string& features_path, const std::string& lables_path) {
    auto features = read_mnist_images(features_path);
    auto lables = read_mnist_labels(lables_path);
    if (features.size() != lables.size()) {
        throw std::logic_error("features and lables have different sizes");
    }

    std::vector<DataSample> data(features.size());
    for (size_t i = 0; i < data.size(); ++i) {
        data[i].first.resize(features[i].size());
        for (size_t f_id = 0; f_id < features[i].size(); ++f_id) {
            data[i].first[f_id] = static_cast<double>(features[i][f_id]);
        }

        data[i].second.resize(10, 0.0);
        data[i].second[lables[i]] = 1.0;
    }

    return data;
}
