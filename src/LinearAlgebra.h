#pragma once

#include <Eigen/Core>

namespace NeuralNetwork {

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
using Index = Eigen::Index;
using RowVector = Eigen::RowVectorXd;

using ConvertToVector = Eigen::Map<Vector, Eigen::Unaligned>;

};  // namespace NeuralNetwork
