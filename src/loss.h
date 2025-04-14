#pragma once

#include "LinearAlgebra.h"

#include <functional>

namespace NeuralNetwork {

enum class LossType { Mse, CrossEntropy };

template <LossType type>
struct LossData;

template <>
struct LossData<LossType::Mse> {
    static double loss(const Vector& predicted, const Vector& target) {
        assert(predicted.size() == target.size());
        return (predicted - target).array().square().mean();
    }

    static RowVector gradient(const Vector& predicted, const Vector& target) {
        assert(predicted.size() == target.size());
        return ((predicted - target).array() * 2.0 / predicted.size()).matrix().transpose();
    }
};

template <>
struct LossData<LossType::CrossEntropy> {
    static double loss(const Vector& predicted, const Vector& target) {
        assert(predicted.size() == target.size());
        auto p_log = predicted.array().log();
        return -(p_log * target.array()).sum();
    }

    static RowVector gradient(const Vector& predicted, const Vector& target) {
        assert(predicted.size() == target.size());
        return (predicted - target).transpose();
    }
};

struct LossSelector {
    using LossFunctionType = double(const Vector&, const Vector&);
    using GradienFunctionType = RowVector(const Vector&, const Vector&);

    static LossFunctionType* loss(LossType type) {
        LossFunctionType* ptr;
        switch (type) {
            case LossType::Mse:
                ptr = &LossData<LossType::Mse>::loss;
                break;
            case LossType::CrossEntropy:
                ptr = &LossData<LossType::CrossEntropy>::loss;
                break;
            default:
                assert(false && "Usupported LossType");
                break;
        }

        return ptr;
    }

    static GradienFunctionType* gradient(LossType type) {
        GradienFunctionType* ptr;
        switch (type) {
            case LossType::Mse:
                ptr = &LossData<LossType::Mse>::gradient;
                break;
            case LossType::CrossEntropy:
                ptr = &LossData<LossType::CrossEntropy>::gradient;
                break;
            default:
                assert(false && "Usupported LossType");
                break;
        }

        return ptr;
    }
};

class LossFunction {
public:
    using LossFunctionType = LossSelector::LossFunctionType;
    using GradienFunctionType = LossSelector::GradienFunctionType;

    LossFunction(LossType type);
    LossFunction(LossFunctionType loss, GradienFunctionType gradient);

    ~LossFunction() = default;

    double Loss(const Vector& predicted, const Vector& target) const;

    RowVector Gradient(const Vector& predicted, const Vector& target) const;

    static LossFunction Mse();
    static LossFunction CrossEntropy();

private:
    std::function<LossFunctionType> loss_;
    std::function<GradienFunctionType> gradient_;
};

};  // namespace NeuralNetwork
