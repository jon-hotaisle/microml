#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <cmath>
#include <functional>
#include <iostream>
#include "prime.hpp"
#include "loss.hpp"
#include "prime.hpp"
#include <algorithm> 



float CrossEntropyLoss::grad_calc_local() {
    float t = target -> get_val();
    float p = pred -> get_val(); 
    float grad = (-t/(p+eps)) + ((1-t)/(1-p+eps));
    return grad;
}

ValuePtr CrossEntropyLoss::forward() {
    float t = target->get_val();
    float p = pred->get_val();
    float loss_val = - (t * log(p + eps)+(1 - t)*log(1 - p + eps));

    auto loss_node = Value::create(loss_val, "bce");
    ValuePtr T = target, P = pred, L = loss_node;
    float eps = this->eps;  
    loss_node->_backward = [T, P, L, eps]() {
        float t = T->get_val();
        float p = P->get_val();
        float grad_local = (-t/(p+eps)) + ((1-t)/(1-p+eps));
        P->add_grad(grad_local * L->get_grad());
    };


    loss_node->set_prev(vector<ValuePtr>{target, pred});
    
    return loss_node;
}


float MSELoss::grad_calc_local() {
    float t = target -> get_val();
    float p = pred -> get_val(); 
    float grad = 2*(p-t);
    return grad;
}

ValuePtr MSELoss::forward() {
    float t = target -> get_val();
    float p = pred -> get_val();
    float loss_val = pow(p-t, 2);
    auto loss_node = Value::create(loss_val, "mse");
    ValuePtr T = target, P = pred, L = loss_node;
    loss_node->_backward = [T, P, L]() {
        float grad_local = 2.0f * (P->get_val() - T->get_val());
        P->add_grad(grad_local * L->get_grad());
    };
    loss_node->set_prev(vector<ValuePtr>{target, pred});
    
    return loss_node;
}


float ReLU::grad_calc_local() {
    float inp = input->get_val();
    float grad = (inp > 0.0f ? 1.0f : 0.0f);
    return grad;
}

ValuePtr ReLU::forward() {
    auto activation_node = Value::create(max((float) 0.0, input->get_val()), "relu");
    activation_node->_backward = [activation_node, this]() {
        float grad = grad_calc_local();
        input->add_grad(grad * activation_node->get_grad());
    };
    activation_node->set_prev(vector<ValuePtr>{input});
    return activation_node;
}