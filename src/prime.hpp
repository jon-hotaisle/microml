#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <cmath>
#include <functional>
#include <iostream>
#include <unordered_set>
#include "mat.hpp"
#pragma once

using namespace std;

class Value;
using ValuePtr = std::shared_ptr<Value>;

class Value : public std::enable_shared_from_this<Value> {
private:
    inline static size_t currentID = 0;
    Tensor data;
    Tensor grad;
    std::string op;
    size_t id;
    std::vector<ValuePtr> prev;
    // void run_backward(vector<ValuePtr>& topo);
    
public:
    Value(float data, const string &op, size_t id);
    Value(const Tensor &t, const string &op, size_t id);

    ~Value();

    static ValuePtr create(float data, const std::string& op = "");
    static ValuePtr create(const Tensor &t, const std::string &op="");

    void print(bool verbose = true, int depth = 0);

    float get_val();
    void set_val(float val);
    float get_grad();
    void set_grad(float g);
    void add_grad(float g);
    string get_op();
    void set_prev(const vector<ValuePtr>& parents);


    static ValuePtr add(const ValuePtr& lhs, const ValuePtr& rhs);
    static ValuePtr sub(const ValuePtr& lhs, const ValuePtr& rhs);
    static ValuePtr mult(const ValuePtr& lhs, const ValuePtr& rhs);
    static ValuePtr exp(const ValuePtr& base, const ValuePtr& power);
    static ValuePtr div(const ValuePtr& num, const ValuePtr& den);
    static ValuePtr divp(const ValuePtr& num, const ValuePtr& den);



    function<void()> _backward;
    void backward();
    void topo_sort(vector<ValuePtr>& topo, unordered_set<Value*>& visited);
    static void dump_to_dot(const vector<ValuePtr>& topo, const string& filename);
    static void visualize(const std::vector<ValuePtr>& topo, const std::string& basename);
    
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
