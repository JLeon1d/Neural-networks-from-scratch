#pragma once

#include "LinearAlgebra.h"

#include <utility>

namespace NeuralNetwork {

struct DataSample {
    Vector features;
    Vector target;
};

using DataBatch = std::vector<DataSample>;

class DataLoader {
public:
    using LoadFunctionType = std::vector<std::pair<std::vector<double>, std::vector<double>>>(const std::string&,
                                                                                              const std::string&);

    template <typename F>
    DataLoader(const std::string& features_path, const std::string& lables_path, F f)
        : features_path_(features_path), lables_path_(lables_path), f_(std::move(f)) {
    }

    DataBatch Load();

private:
    std::string features_path_;
    std::string lables_path_;

    std::function<LoadFunctionType> f_;
};

};  // namespace NeuralNetwork
