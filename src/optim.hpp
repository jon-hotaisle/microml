#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <cmath>
#include <functional>
#include <iostream>
#include <unordered_set>
#include "prime.hpp"

class AdamW {
    private:
        float lr; 
        float b1;
        float b2;
        float weight_decay;
        float epsilon;
        float m_t;
        float v_t;
        int t;
        ValuePtr param;
    
    public:
    
        AdamW(float lr, ValuePtr param, float b1 = 0.9, float b2 = 0.999, float epsilon = 1e-8, float weight_decay=0.01) {
                this->lr = lr;
                this->b1 = b1;
                this->b2 = b2;
                this->param = param;
                this->m_t = 0;
                this->v_t = 0;
                this->t = 0;
            }

        void step();
};
