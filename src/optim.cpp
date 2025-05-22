#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <cmath>
#include <functional>
#include <iostream>
#include <unordered_set>
#include "prime.hpp"
#include "optim.hpp"

void AdamW::step() {
    m_t = b1 * m_t + (1-b1) * param->get_grad();
    v_t = b2 * v_t + (1-b2) * pow((param->get_grad()), 2);
    t++;

    float m_tb = m_t/(1-pow(b1, t));
    float v_tb = v_t/(1-pow(b2, t));

    float update = (m_tb / (std::sqrt(v_tb) + epsilon)) + weight_decay * param->get_val();

    float new_val = param->get_val() - lr * update;
    param->set_val(new_val);
    
    param->set_grad(0.0f);
}