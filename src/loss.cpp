#include "loss.hpp"
#include "prime.hpp"
#include <cmath>
#include <algorithm>
using namespace std;


CrossEntropyLoss::CrossEntropyLoss(ValuePtr tgt, ValuePtr pr, size_t n_classes):
    target(move(tgt)), pred(move(pr)), eps(numeric_limits<float>::epsilon()), n_classes(n_classes)
{}

float CrossEntropyLoss::grad_calc_local() const {
    if (n_classes == 2) {
        float t = target->get_val();
        float p = pred->get_val();
        return (-t / (p + eps)) + ((1 - t) / (1 - p + eps));
    } else {
        return 0.0f; // not used in multi-class case
    }
}

ValuePtr CrossEntropyLoss::forward() {
    if (n_classes == 2) {
        const Tensor& target_tensor = target->get_tensor();
        const Tensor& pred_tensor = pred->get_tensor();
        
        float t, p;
        if (target_tensor.is_scalar()) {
            t = target_tensor.scalar_value();
        } else {
            t = target_tensor.data[0];  
        }
        
        if (pred_tensor.is_scalar()) {
            p = pred_tensor.scalar_value();
        } else {
            p = pred_tensor.data[0]; 
        }
        
        float loss_val = -(t * log(p + eps) + (1 - t) * log(1 - p + eps));
        auto loss_node = Value::create(loss_val, "bce");
        
        ValuePtr T = target, P = pred, L = loss_node;
        float local_eps = eps;
        
        loss_node->_backward = [T, P, L, local_eps]() {
            const Tensor& target_tensor = T->get_tensor();
            const Tensor& pred_tensor = P->get_tensor();
            const Tensor& loss_grad = L->get_grad();
            
            float t = target_tensor.is_scalar() ? target_tensor.scalar_value(): target_tensor.data[0];
            float p = pred_tensor.is_scalar() ? pred_tensor.scalar_value(): pred_tensor.data[0];
            float loss_grad_val = loss_grad.is_scalar() ? loss_grad.scalar_value(): loss_grad.data[0];
            
            float grad_val = ((-t / (p + local_eps)) + ((1 - t) / (1 - p + local_eps))) * loss_grad_val;
            
            if (pred_tensor.is_scalar()) {
                P->add_grad(Tensor(grad_val));
            } else {
                Tensor grad_tensor = Tensor::zeros(pred_tensor.shape());
                grad_tensor.data[0] = grad_val;
                P->add_grad(grad_tensor);
            }
        };

        loss_node->set_prev({target, pred});
        return loss_node;
        
    } else {
        const Tensor& logits_tensor = pred->get_tensor();
        const Tensor& target_tensor = target->get_tensor();
        
        // softmax
        float max_val = *std::max_element(logits_tensor.data.begin(), logits_tensor.data.end());
        
        vector<float> softmax_vals(logits_tensor.data.size());
        float sum_exp = 0.0f;
        
        for (size_t i = 0; i < logits_tensor.data.size(); ++i) {
            softmax_vals[i] = std::exp(logits_tensor.data[i] - max_val);
            sum_exp += softmax_vals[i];
        }
        
        for (size_t i = 0; i < softmax_vals.size(); ++i) {
            softmax_vals[i] /= sum_exp;
        }
        
        float loss_val = 0.0f;
        for (size_t i = 0; i < target_tensor.data.size(); ++i) {
            if (target_tensor.data[i] > 0.0f) {
                loss_val -= target_tensor.data[i] * std::log(softmax_vals[i] + eps);
            }
        }
        
        auto loss_node = Value::create(loss_val, "cross_entropy");
        
        ValuePtr T = target, P = pred, L = loss_node;
        float local_eps = eps;
        
        loss_node->_backward = [T, P, L, local_eps]() {
            const Tensor& logits_tensor = P->get_tensor();
            const Tensor& target_tensor = T->get_tensor();
            const Tensor& loss_grad = L->get_grad();
            
            float max_val = *std::max_element(logits_tensor.data.begin(), logits_tensor.data.end());
            vector<float> softmax_vals(logits_tensor.data.size());
            float sum_exp = 0.0f;
            
            for (size_t i = 0; i < logits_tensor.data.size(); ++i) {
                softmax_vals[i] = std::exp(logits_tensor.data[i] - max_val);
                sum_exp += softmax_vals[i];
            }
            
            for (size_t i = 0; i < softmax_vals.size(); ++i) {
                softmax_vals[i] /= sum_exp;
            }
            
            Tensor pred_grad = Tensor::zeros(logits_tensor.shape());
            float loss_grad_val = loss_grad.is_scalar() ? loss_grad.scalar_value() : loss_grad.data[0];
            
            for (size_t i = 0; i < softmax_vals.size(); ++i) {
                pred_grad.data[i] = (softmax_vals[i] - target_tensor.data[i]) * loss_grad_val;
            }
            
            P->add_grad(pred_grad);
        };
        
        loss_node->set_prev({target, pred});
        return loss_node;
    }
}




MSELoss::MSELoss(ValuePtr tgt, ValuePtr pr):
    target(move(tgt)), pred(move(pr))
{}

float MSELoss::grad_calc_local() const {
    const Tensor& target_tensor = target->get_tensor();
    const Tensor& pred_tensor = pred->get_tensor();
    
    float t = target_tensor.is_scalar() ? target_tensor.scalar_value() : target_tensor.data[0];
    float p = pred_tensor.is_scalar() ? pred_tensor.scalar_value() : pred_tensor.data[0];
    
    return 2.0f * (p - t);
}

ValuePtr MSELoss::forward() {
    const Tensor& target_tensor = target->get_tensor();
    const Tensor& pred_tensor = pred->get_tensor();
    
    float t, p;
    if (target_tensor.is_scalar()) {
        t = target_tensor.scalar_value();
    } else {
        t = target_tensor.data[0]; 
    }
    
    if (pred_tensor.is_scalar()) {
        p = pred_tensor.scalar_value();
    } else {
        p = pred_tensor.data[0];  
    }
    
    float loss_val = (p - t) * (p - t);
    auto loss_node = Value::create(loss_val, "mse");

    ValuePtr T = target, P = pred, L = loss_node;
    loss_node->_backward = [T, P, L]() {
        const Tensor& target_tensor = T->get_tensor();
        const Tensor& pred_tensor = P->get_tensor();
        const Tensor& loss_grad = L->get_grad();
        
        float t = target_tensor.is_scalar() ? target_tensor.scalar_value(): target_tensor.data[0];
        float p = pred_tensor.is_scalar() ? pred_tensor.scalar_value(): pred_tensor.data[0];
        float loss_grad_val = loss_grad.is_scalar() ? loss_grad.scalar_value(): loss_grad.data[0];
        
        float grad_val = 2.0f * (p - t) * loss_grad_val;
        
        if (pred_tensor.is_scalar()) {
            P->add_grad(Tensor(grad_val));
        } else {
            Tensor grad_tensor = Tensor::zeros(pred_tensor.shape());
            grad_tensor.data[0] = grad_val;
            P->add_grad(grad_tensor);
        }
    };

    loss_node->set_prev({target, pred});
    return loss_node;
}


ReLU::ReLU(ValuePtr inp)
  : input(move(inp))
{}

float ReLU::grad_calc_local() const {
    return input->get_val() > 0.0f ? 1.0f: 0.0f;
}

Tensor ReLU::grad_calc_local_tensor() const {
    Tensor input_tensor = input->get_tensor();
    return input_tensor.apply([](float x) { 
        return x > 0.0f ? 1.0f: 0.0f; 
    });
}

ValuePtr ReLU::forward() {
    if (input->get_tensor().is_scalar()) {
        float x = input->get_val();
        auto node = Value::create(max(0.0f, x), "relu");
        ValuePtr I = input, N = node;
        node->_backward = [I, N]() {
            Tensor node_grad = N->get_grad();
            float inp_grad = I->get_val() > 0.0f ? 1.0f: 0.0f;
            I->add_grad(node_grad * Tensor(inp_grad));
        };
        node->set_prev({input});
        return node;
    } else {
        Tensor input_tensor = input->get_tensor();
        Tensor output_tensor = input_tensor.relu();
        
        auto node = Value::create(output_tensor, "relu");
        node->set_prev({input});
        
        ValuePtr I = input, N = node;
        node->_backward = [I, N]() {
            Tensor node_grad = N->get_tensor_grad();
            Tensor input_tensor = I->get_tensor();
            
            Tensor input_grad = input_tensor.apply([](float x) { 
                return x > 0.0f ? 1.0f: 0.0f; 
            });
            
            Tensor final_grad = node_grad * input_grad;
            I->add_grad(final_grad);
        };
        
        return node;
    }

}
