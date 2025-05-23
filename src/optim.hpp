#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <cmath>
#include <functional>
#include <iostream>
#include <unordered_set>
#include "prime.hpp"
using namespace std;

class AdamW {
    private:
        float lr; 
        float b1;
        float b2;
        float weight_decay;
        float epsilon;
        Tensor m_t; 
        Tensor v_t; 
        int t;
        ValuePtr param;
        bool initialized;  
        
        void initialize_momentum_tensors();
    
    public:
        AdamW(float lr, ValuePtr param, float b1 = 0.9, float b2 = 0.999, 
              float epsilon = 1e-8, float weight_decay = 0.01) 
            : lr(lr), b1(b1), b2(b2), weight_decay(weight_decay), 
              epsilon(epsilon), param(param), t(0), initialized(false) {}

        void step();
};