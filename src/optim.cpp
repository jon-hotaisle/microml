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
using namespace std;

void AdamW::step() {
    float g = param->get_grad().scalar_value();

    m_t = b1 * m_t + (1 - b1) * g;
    v_t = b2 * v_t + (1 - b2) * g * g;
    t++;

    float m_hat = m_t / (1 - std::pow(b1, t));
    float v_hat = v_t / (1 - std::pow(b2, t));

    float update = m_hat / (std::sqrt(v_hat) + epsilon)
                   + weight_decay * param->get_val();

    float new_val = param->get_val() - lr * update;
    param->set_val(new_val);

    param->set_grad(Tensor(0.0f));

}