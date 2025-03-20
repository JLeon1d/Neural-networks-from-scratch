#pragma once

#include "LinearAlgebra.h"
#include <utility>

namespace NeuralNetwork {

struct DataSample {
    Vector features;
    Vector target;
};

using DataBatch = std::vector<DataSample>;

// all features have the same type ?
class DataLoader {
public:
    using LoadFunction = std::vector<std::pair<std::vector<double>, std::vector<double>>>(const std::string&, const std::string&);

    DataLoader(const std::string& features_path, const std::string& lables_path, LoadFunction f);

    DataBatch Load();

private:
    std::string features_path_;
    std::string lables_path_;

    std::function<LoadFunction> f_;
};

}; // namespace NeuralNetwork