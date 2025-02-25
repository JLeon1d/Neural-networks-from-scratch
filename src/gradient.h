#pragma once

#include "Eigen/Core"
#include "AnyMovable.h"

#include <initializer_list>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace NeuralNetwork {

using GradientPair = std::pair<MatrixXd, VectorXd>;
using FuncType = GradientPair(const VectorXd&, const MatrixXd&, double);

class ClassicGradient {
public:
    explicit ClassicGradient() = default;

    ClassicGradient(const ClassicGradient& oth) = delete;

    ClassicGradient(ClassicGradient&& oth) = default;

    ClassicGradient& operator=(const ClassicGradient& oth) = delete;

    ClassicGradient& operator=(ClassicGradient&& oth) = default;

    GradientPair operator()(const VectorXd& x, const MatrixXd& u, double learning_rate);
private:
};

class MomentumGradient {
public:
    static constexpr double kDefaultAlpha = 0.9;

    explicit MomentumGradient(size_t x_size, size_t u_size, double alpha = kDefaultAlpha);

    MomentumGradient(const MomentumGradient& oth) = delete;

    MomentumGradient(MomentumGradient&& oth) = default;

    MomentumGradient& operator=(const MomentumGradient& oth) = delete;

    MomentumGradient& operator=(MomentumGradient&& oth) = default;

    GradientPair operator()(const VectorXd& x, const MatrixXd& u, double learning_rate);

    ~MomentumGradient() = default;

private:
    MatrixXd momentum_A_;
    VectorXd momentum_b_;

    double alpha_;
};

class AdaGradient {
public:
    static constexpr double epsilon = 1e-8;

    explicit AdaGradient(size_t x_size, size_t u_size);

    AdaGradient(const AdaGradient& oth) = delete;

    AdaGradient(AdaGradient&& oth) = default;

    AdaGradient& operator=(const AdaGradient& oth) = delete;
    
    AdaGradient& operator=(AdaGradient&& oth) = default;
    
    GradientPair operator()(const VectorXd& x, const MatrixXd& u, double learning_rate);
    
    ~AdaGradient() = default;
private:
    MatrixXd A_g_;
    VectorXd b_g_;
};

class RMSPropGradient {
public:
    static constexpr double kDefaultAlpha = 0.9;
    static constexpr double epsilon = 1e-8;

    explicit RMSPropGradient(size_t x_size, size_t u_size, double alpha = kDefaultAlpha);
    
    RMSPropGradient(const RMSPropGradient& oth) = delete;
    
    RMSPropGradient(RMSPropGradient&& oth) = default;
    
    RMSPropGradient& operator=(const RMSPropGradient& oth) = delete;
    
    RMSPropGradient& operator=(RMSPropGradient&& oth) = default;
    
    GradientPair operator()(const VectorXd& x, const MatrixXd& u, double learning_rate);
    
    ~RMSPropGradient() = default;
private:
    MatrixXd A_g_;
    VectorXd b_g_;
    
    double alpha_;
};

// maybe make bias-correction (alpha ^ step) optional ?
class AdamGradient {
public:
    static constexpr double kDefaultAlpha = 0.9;
    static constexpr double epsilon = 1e-8;
    
    explicit AdamGradient(size_t x_size, size_t u_size, double alpha_linear = kDefaultAlpha,
                          double alpha_square = kDefaultAlpha);
        
    AdamGradient(const AdamGradient& oth) = delete;
        
    AdamGradient(AdamGradient&& oth) = default;
        
    AdamGradient& operator=(const AdamGradient& oth) = delete;
        
    AdamGradient& operator=(AdamGradient&& oth) = default;
        
    GradientPair operator()(const VectorXd& x, const MatrixXd& u, double learning_rate);
        
    ~AdamGradient() = default;
private:
    MatrixXd A_linear_;
    MatrixXd A_square_;
    VectorXd b_linear_;
    VectorXd b_square_;

    double alpha_linear_;
    double alpha_square_;

    double alpha_linear_step_;  // alpha_linear_ ^ step
    double alpha_square_step_;  // alpha_square_ ^ step
};

template<class TBase>
class IGradientFunction : public TBase {
public:
    virtual GradientPair operator()(const VectorXd& x, const MatrixXd& u, double learning_rate) = 0;
};

template<class TBase, class TObject>
class CGradientFunctionImpl : public TBase {
    using CBase = TBase;
public:
    using CBase::CBase;

    GradientPair operator()(const VectorXd& x, const MatrixXd& u, double learning_rate) override {
        return CBase::Object()(x, u, learning_rate);
    }
};

class GradientFunction : public NSLibrary::CAnyMovable<IGradientFunction, CGradientFunctionImpl> {
    using CBase = NSLibrary::CAnyMovable<IGradientFunction, CGradientFunctionImpl>;
public:
    using CBase::CBase;

    enum Type {
        Classic,
        Momentum,
        AdaGrad,
        RMSProp,
        Adam
    };

    struct Initializer {
        Type type;
        std::initializer_list<double> args;
    };
};

}; // namespace NeuralNetwork