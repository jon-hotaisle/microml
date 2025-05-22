#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <cmath>
#include <functional>
#include <iostream>
#include "prime.hpp"
#include <limits>


class CrossEntropyLoss {        
    private:
        ValuePtr target;
        ValuePtr pred;
        float eps;

    public:
        CrossEntropyLoss(ValuePtr target, ValuePtr pred) {
            this->target = target;
            this->pred = pred;
            this->eps = numeric_limits<float>::epsilon();
        
        }
        ValuePtr forward();
        float grad_calc_local();
        function<void()> _backward;
};

class MSELoss {        
    private:
        ValuePtr target;
        ValuePtr pred;

    public:
        MSELoss(ValuePtr target, ValuePtr pred) {
            this->target = target;
            this->pred = pred;        
        }
        ValuePtr forward();
        float grad_calc_local();
        function<void()> _backward;
};


class ReLU {        
    private:
        ValuePtr input;

    public:
        ReLU(ValuePtr input) {
            this->input = input;        
        }
        ValuePtr forward();
        float grad_calc_local();
        function<void()> _backward;
};