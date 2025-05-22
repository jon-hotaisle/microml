#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <cmath>
#include <functional>
#include <iostream>
#include <unordered_set>
#pragma once

using namespace std;
class Value;
using ValuePtr = std::shared_ptr<Value>;

class Value : public std::enable_shared_from_this<Value> {
private:
    inline static size_t currentID = 0;
    float data;
    float grad;
    std::string op;
    size_t id;
    std::vector<ValuePtr> prev;
    // void run_backward(vector<ValuePtr>& topo);
    
public:
    Value(float data, const std::string &op, size_t id);
    ~Value();

    static ValuePtr create(float data, const std::string& op = "");

    void print(bool verbose = true, int depth = 0);

    float get_val();
    void set_val(float val);
    void set_prev(const vector<ValuePtr>& parents);
    void set_grad(float grad);
    std::string get_op();

    float get_grad();
    void add_grad(float g);


    static ValuePtr add(const ValuePtr& lhs, const ValuePtr& rhs);
    static ValuePtr sub(const ValuePtr& lhs, const ValuePtr& rhs);
    static ValuePtr mult(const ValuePtr& lhs, const ValuePtr& rhs);
    static ValuePtr exp(const ValuePtr& base, const ValuePtr& power);

    function<void()> _backward;
    void backward();
    void topo_sort(vector<ValuePtr>& topo, unordered_set<Value*>& visited);
    void dump_to_dot(const vector<ValuePtr>& topo, const string& filename);
    
};