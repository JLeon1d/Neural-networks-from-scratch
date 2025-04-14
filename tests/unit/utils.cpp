#include "utils.h"
#include "LinearAlgebra.h"

namespace Utils {

NeuralNetwork::Vector MakeVector(const std::initializer_list<double>& values) {
    return Eigen::Map<const NeuralNetwork::Vector>(values.begin(), values.size());
}

NeuralNetwork::RowVector MakeRowVector(const std::initializer_list<double>& values) {
    return Eigen::Map<const NeuralNetwork::RowVector>(values.begin(), values.size());
}

};  // namespace Utils
