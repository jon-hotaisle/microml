#pragma once

#include <functional>
#include <limits>
#include "prime.hpp"
using namespace std;

class CrossEntropyLoss {
    private:
        ValuePtr target;
        ValuePtr pred;
        float eps;

    public:
        CrossEntropyLoss(ValuePtr target, ValuePtr pred);
        ValuePtr forward();
        float grad_calc_local() const;
};

class MSELoss {
    private:
        ValuePtr target;
        ValuePtr pred;

    public:
        MSELoss(ValuePtr target, ValuePtr pred);
        ValuePtr forward();
        float grad_calc_local() const;
};

class ReLU {
    private:
        ValuePtr input;

    public:
        explicit ReLU(ValuePtr input);
        ValuePtr forward();
        float grad_calc_local() const;
        Tensor grad_calc_local_tensor() const;
};
