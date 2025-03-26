#include <iostream>
#include "LinearAlgebra.h"
#include "AnyMovable.h"

class A {
public:
    void print(int x) {
        std::cout << x;
    }
};

class B {
public:
    void print(int x) {
        std::cout << x + 1;
    }
};

template <class TBase>
class IOptimizer : public TBase {
public:
    virtual void Print(int x) = 0;
};

template <class TBase, class TObject>
class COptimizerImpl : public TBase {
    using CBase = TBase;

public:
    using CBase::CBase;

    void Print(int x) override {
        return CBase::Object().print(x);
    }
};

class Beb : public NSLibrary::CAny<IOptimizer, COptimizerImpl> {
    using CBase = NSLibrary::CAny<IOptimizer, COptimizerImpl>;

public:
    using CBase::CBase;
};

int main() {
    Beb x;
    x = std::move(A());
    x = B();
}
