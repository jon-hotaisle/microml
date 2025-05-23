#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <cmath>
#include <functional>
#include <iostream>
#include <unordered_set>
#include "tensor.hpp"
#pragma once

using namespace std;

class Value;
using ValuePtr = shared_ptr<Value>; // ---- Credit to Gautam Sharma for inspiring this initial Value and ValuePtr setup (https://www.youtube.com/watch?v=wOuMVD3XoHU) ---

class Value : public enable_shared_from_this<Value> {
private:
    inline static size_t currentID = 0;
    Tensor data;
    Tensor grad;
    string op;
    size_t id;
    vector<ValuePtr> prev;
    // void run_backward(vector<ValuePtr>& topo);
    
public:
    Value(float data, const string &op, size_t id);
    Value(const Tensor &t, const string &op, size_t id);

    ~Value();

    static ValuePtr create(float data, const string& op = "");
    static ValuePtr create(const Tensor &t, const string &op="");

    void print(bool verbose = true, int depth = 0);

    float get_val() const;
    const Tensor& get_grad() const;
    void set_grad(const Tensor &g);
    void set_grad(float g);

    void add_grad(const Tensor &g);
    string get_op() const;

    void set_val(float val);

    const Tensor& get_tensor() const;
    const Tensor& get_tensor_grad() const;
    void set_tensor_grad(const Tensor& g);
    void add_tensor_grad(const Tensor& g);
    void set_tensor(const Tensor& t);

    // float get_grad();
    // void set_grad(float g);
    // void add_grad(float g);


    void set_prev(const vector<ValuePtr>& parents);


    static ValuePtr add(const ValuePtr& lhs, const ValuePtr& rhs);
    static ValuePtr sub(const ValuePtr& lhs, const ValuePtr& rhs);
    static ValuePtr mult(const ValuePtr& lhs, const ValuePtr& rhs);
    static ValuePtr exp(const ValuePtr& base, const ValuePtr& power);
    static ValuePtr div(const ValuePtr& num, const ValuePtr& den);
    static ValuePtr divp(const ValuePtr& num, const ValuePtr& den);
    static ValuePtr matmul(const ValuePtr& A, const ValuePtr& B);



    function<void()> _backward;
    void backward(bool retain_graph);
    void topo_sort(vector<ValuePtr>& topo, unordered_set<Value*>& visited);
    static void dump_to_dot(const vector<ValuePtr>& topo, const string& filename);
    static void visualize(const vector<ValuePtr>& topo, const string& basename);
    
};


//// ------ operator overloading support ---- ////

inline ValuePtr operator+(const ValuePtr& a, const ValuePtr& b) {
  return Value::add(a, b);
}

inline ValuePtr operator-(const ValuePtr& a, const ValuePtr& b) {
  return Value::sub(a, b);
}

inline ValuePtr operator*(const ValuePtr& a, const ValuePtr& b) {
  return Value::mult(a, b);
}

inline ValuePtr operator/(const ValuePtr& a, const ValuePtr& b) {
  return Value::divp(a, b);
}

inline ValuePtr operator+(const ValuePtr& a, float b) {
  return a + Value::create(b);
}
inline ValuePtr operator-(const ValuePtr& a, float b) {
  return a - Value::create(b);
}
inline ValuePtr operator*(const ValuePtr& a, float b) {
  return a * Value::create(b);
}
inline ValuePtr operator/(const ValuePtr& a, float b) {
  return a / Value::create(b);
}

inline ValuePtr operator+(float a, const ValuePtr& b) {
  return Value::create(a) + b;
}
inline ValuePtr operator-(float a, const ValuePtr& b) {
  return Value::create(a) - b;
}
inline ValuePtr operator*(float a, const ValuePtr& b) {
  return Value::create(a) * b;
}
inline ValuePtr operator/(float a, const ValuePtr& b) {
  return Value::create(a) / b;
}

namespace std {
    inline ValuePtr pow(const ValuePtr& base, const ValuePtr& exp) { 
        return Value::exp(base, exp); 
    }
    inline ValuePtr pow(const ValuePtr& base, float p) { 
        return pow(base, Value::create(p)); 
    }
    inline ValuePtr pow(float b, const ValuePtr& p) { 
        return pow(Value::create(b), p); 
    }
}
