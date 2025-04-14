#pragma once

#include "LinearAlgebra.h"

namespace Utils {

NeuralNetwork::Vector MakeVector(const std::initializer_list<double>& values);

NeuralNetwork::RowVector MakeRowVector(const std::initializer_list<double>& values);

};  // namespace Utils
