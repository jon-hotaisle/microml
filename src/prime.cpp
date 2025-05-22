#include "prime.hpp"
#include "loss.hpp"
#include <iostream>
#include <cmath>
#include <unordered_set>

Value::Value(float data, const string &op, size_t id)
    : data(data), grad(0.0), op(op), id(id) {}

Value::~Value() {
    --Value::currentID;
}

ValuePtr Value::create(float data, const string& op) {
    return make_shared<Value>(data, op, Value::currentID++);
}

void Value::print(bool verbose, int depth) {
    for (int i = 0; i < depth; ++i) {
        cout << "     ";
    }
    
    cout << "[data=" << data 
              << ", grad=" << grad 
              << ", op=" << op 
              << ", id=" << id 
              << "]\n";

    if (verbose && !prev.empty()) {
        for (const auto& child : prev) {
            if (child) {
                child->print(verbose, depth + 1);
            }
        }
    }
}

float Value::get_val() {
    return data;
}

void Value::set_prev(const vector<ValuePtr>& parents) {
    this->prev = parents;
}

void Value::set_grad(float grad) {
    this->grad = grad;
}

void Value::add_grad(float g) {
    this->grad += g;
}

float Value::get_grad() {
    return this->grad;
}


string Value::get_op() {
    return op;
}

ValuePtr Value::add(const ValuePtr& lhs, const ValuePtr& rhs) {
    auto out = lhs->data + rhs->data;
    auto node = Value::create(out, "+");
    node->_backward = [lhs, rhs, node](){
        lhs->grad += (float) 1 * node->grad;
        rhs->grad += (float) 1 * node->grad;
    };
    node->set_prev(vector<ValuePtr>{lhs, rhs});
    return node;
}

ValuePtr Value::sub(const ValuePtr& lhs, const ValuePtr& rhs) {
    auto out = lhs->data - rhs->data;
    auto node = Value::create(out, "-");
    node->_backward = [lhs, rhs, node](){
        lhs->grad += (float) 1 * node->grad;
        rhs->grad += (float) -1 * node->grad;
    };
    node->set_prev(vector<ValuePtr>{lhs, rhs});
    return node;
}

ValuePtr Value::mult(const ValuePtr& lhs, const ValuePtr& rhs) {
    auto out = lhs->data * rhs->data;
    auto node = Value::create(out, "*");
    node->_backward = [lhs, rhs, node](){
        lhs->grad += rhs->get_val() * node->grad;
        rhs->grad += lhs->get_val() * node->grad;
    };
    node->set_prev(vector<ValuePtr>{lhs, rhs});
    return node;
}


void Value::backward() {
    vector<ValuePtr> topo;
    unordered_set<Value*> visited;
    topo_sort(topo, visited);
    this->grad = 1.0f;

    for (auto next = topo.rbegin(); next != topo.rend(); ++next) {
        auto& node = *next;
        if (node->_backward) {
            node->_backward();
        }
    }
}

void Value::topo_sort(vector<ValuePtr>& topo, unordered_set<Value*>& visited) {
    if (visited.count(this)) {
        return;
    }
    visited.insert(this);
    for (auto& p : prev) {
        p->topo_sort(topo, visited);
    }
    topo.push_back(shared_from_this());
}


