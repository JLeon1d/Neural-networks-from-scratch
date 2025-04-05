#include "data_loader.h"

namespace NeuralNetwork {

DataBatch DataLoader::Load() {
    assert(f_);
    auto raw_data = f_(features_path_, lables_path_);
    if (raw_data.empty()) {
        throw std::logic_error("empty data");
    }

    size_t features_size = raw_data[0].first.size();
    size_t lables_size = raw_data[0].second.size();

    DataBatch data(raw_data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        if (raw_data[i].first.size() != features_size) {
            throw std::logic_error("wrong featires size");
        }

        data[i].features = ConvertToVector(raw_data[i].first.data(), features_size);
        data[i].target = ConvertToVector(raw_data[i].second.data(), lables_size);
    }

    return data;
}

};  // namespace NeuralNetwork
