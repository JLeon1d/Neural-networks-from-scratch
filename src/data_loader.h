#pragma once

#include <Eigen/Dense>
// #include <type_traits>
#include <utility>

using Eigen::VectorXd;

namespace NeuralNetwork {

using DataSample = std::pair<VectorXd, VectorXd>;
using DataBatch = std::vector<DataSample>;

// all features have the same type ?
// maybe force LoadFuncton to cast to long dobule ?
class DataLoader {
public:
    using LoadFunction = std::vector<std::pair<std::vector<double>, std::vector<double>>>(const std::string&, const std::string&);

    explicit DataLoader(const std::string& features_path, const std::string& lables_path, LoadFunction f);

    DataBatch Load();

private:
    std::string features_path_;
    std::string lables_path_;

    std::function<LoadFunction> f_;
};

}; // namespace NeuralNetwork