#include "loss.hpp"
#include "prime.hpp"
#include <cmath>
#include <algorithm>

CrossEntropyLoss::CrossEntropyLoss(ValuePtr tgt, ValuePtr pr)
  : target(std::move(tgt))
  , pred(std::move(pr))
  , eps(std::numeric_limits<float>::epsilon())
{}

float CrossEntropyLoss::grad_calc_local() const {
    float t = target->get_val();
    float p = pred->get_val();
    return (-t / (p + eps)) + ((1 - t) / (1 - p + eps));
}

ValuePtr CrossEntropyLoss::forward() {
    float t = target->get_val();
    float p = pred->get_val();
    float loss_val = - (t * std::log(p + eps) + (1 - t) * std::log(1 - p + eps));
    auto loss_node = Value::create(loss_val, "bce");

    ValuePtr T = target, P = pred, L = loss_node;
    float local_eps = eps;
    loss_node->_backward = [T, P, L, local_eps]() {
        float t = T->get_val();
        float p = P->get_val();
        float g = (-t / (p + local_eps)) + ((1 - t) / (1 - p + local_eps));
        P->add_grad(g * L->get_grad());
    };

    loss_node->set_prev({target, pred});
    return loss_node;
}

MSELoss::MSELoss(ValuePtr tgt, ValuePtr pr)
  : target(std::move(tgt))
  , pred(std::move(pr))
{}

float MSELoss::grad_calc_local() const {
    float t = target->get_val();
    float p = pred->get_val();
    return 2.0f * (p - t);
}

ValuePtr MSELoss::forward() {
    float t = target->get_val();
    float p = pred->get_val();
    float loss_val = (p - t) * (p - t);
    auto loss_node = Value::create(loss_val, "mse");

    ValuePtr T = target, P = pred, L = loss_node;
    loss_node->_backward = [T, P, L]() {
        float g = 2.0f * (P->get_val() - T->get_val());
        P->add_grad(g * L->get_grad());
    };

    loss_node->set_prev({target, pred});
    return loss_node;
}

ReLU::ReLU(ValuePtr inp)
  : input(std::move(inp))
{}

float ReLU::grad_calc_local() const {
    return input->get_val() > 0.0f ? 1.0f : 0.0f;
}

ValuePtr ReLU::forward() {
    float x = input->get_val();
    auto node = Value::create(std::max(0.0f, x), "relu");
    ValuePtr I = input, N = node;
    node->_backward = [I, N]() {
        float g = N->get_grad();
        float grad_local = I->get_val() > 0.0f ? 1.0f : 0.0f;
        I->add_grad(grad_local * g);
    };
    node->set_prev({input});
    return node;
}
