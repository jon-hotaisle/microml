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

void AdamW::initialize_momentum_tensors() {
    if (!initialized) {
        const auto& param_tensor = param->get_tensor();
        if (param_tensor.is_scalar()) {
            m_t = Tensor(0.0f);
            v_t = Tensor(0.0f);
        } else {
            m_t = Tensor::zeros(param_tensor.shape());
            v_t = Tensor::zeros(param_tensor.shape());
        }
        initialized = true;
    }
}

void AdamW::step() {
    initialize_momentum_tensors();
    
    const Tensor& grad = param->get_grad();
    const Tensor& param_tensor = param->get_tensor();
    
    if (grad.is_scalar()) {
        float g = grad.scalar_value();
        float m_val = b1 * m_t.scalar_value() + (1 - b1) * g;
        float v_val = b2 * v_t.scalar_value() + (1 - b2) * g * g;
        
        m_t = Tensor(m_val);
        v_t = Tensor(v_val);
    } else {

        m_t = m_t * Tensor(b1) + grad * Tensor(1 - b1);
        Tensor grad_squared = grad * grad; 
        v_t = v_t * Tensor(b2) + grad_squared * Tensor(1 - b2);
    }
    
    t++;  
    
    float bias_correction1 = 1.0f - std::pow(b1, t);
    float bias_correction2 = 1.0f - std::pow(b2, t);
    
    if (grad.is_scalar()) {
        float m_hat = m_t.scalar_value() / bias_correction1;
        float v_hat = v_t.scalar_value() / bias_correction2;
        
        float adam_term = m_hat / (std::sqrt(v_hat) + epsilon);
        float weight_decay_term = weight_decay * param_tensor.scalar_value();
        float update = adam_term + weight_decay_term;
        
        float new_val = param_tensor.scalar_value() - lr * update;
        param->set_val(new_val);
    } else {
        Tensor m_hat = m_t * Tensor(1.0f / bias_correction1);
        Tensor v_hat = v_t * Tensor(1.0f / bias_correction2);
        
        Tensor v_hat_sqrt = v_hat.apply([this](float x) { 
            return std::sqrt(x) + epsilon; 
        });
        
        Tensor adam_term = m_hat * v_hat_sqrt.apply([](float x) { 
            return 1.0f / x; 
        });
        
        Tensor weight_decay_term = param_tensor * Tensor(weight_decay);
        Tensor update = adam_term + weight_decay_term;
        Tensor new_param = param_tensor - update * Tensor(lr);
        
        const_cast<Tensor&>(param->get_tensor()) = new_param;
    }
    
    if (grad.is_scalar()) {
        param->set_grad(Tensor(0.0f));
    } else {
        param->set_tensor_grad(Tensor::zeros(grad.shape()));
    }
}