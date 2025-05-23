#include "tensor.hpp"
#include <cassert>
#include <iostream>
#include <algorithm>   
#include <stdexcept> 
#include <cmath>
#include <functional>
#include <random>

using namespace std;

vector<size_t> Tensor::compute_strides(const vector<size_t>& shape) {
    int n = (int)shape.size();
    vector<size_t> s(n, 1);
    for (int k = n - 2; k >= 0; --k) {
        s[k] = s[k+1] * shape[k+1];
    }
    return s;
}

Tensor::Tensor(vector<float> data_, vector<size_t> shape_)
  : shape_(shape_)                    
  , strides_(compute_strides(shape_)) 
  , data(move(data_))           
{
    size_t expected = 1;
    for (auto d : shape_) expected *= d;
    assert(expected == data.size());
}


Tensor::Tensor(float x)
  : data({x})
  , strides_({})
  , shape_({})
{}

size_t Tensor::size() const {
    return data.size();
}

const vector<size_t>& Tensor::shape() const {
    return shape_;
}

size_t Tensor::offset(const vector<size_t>& idx) const {
    assert(idx.size() == shape_.size());
    size_t off = 0;
    for (size_t k = 0; k < idx.size(); ++k) {
        assert(idx[k] < shape_[k]);
        off += idx[k] * strides_[k];
    }
    return off;
}

float& Tensor::at(const vector<size_t>& idx) {
    return data[offset(idx)];
}

const float& Tensor::at(const vector<size_t>& idx) const {
    return data[offset(idx)];
}

float& Tensor::at(size_t i, size_t j) {
    return at({i,j});
}

const float& Tensor::at(size_t i, size_t j) const {
    return at({i,j});
}

bool Tensor::is_scalar() const {
    return shape_.empty();
}

float Tensor::scalar_value() const {
    assert(is_scalar());
    return data[0];
}

Tensor Tensor::transpose() const {
    assert(shape_.size() == 2);
    size_t rows = shape_[0], cols = shape_[1];
    vector<float> new_data(data.size());
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            new_data[j*rows + i] = at(i,j);
        }
    }
    return Tensor(move(new_data), {cols, rows});
}

Tensor Tensor::matmul(const Tensor& x, const Tensor& y) {
    const auto& xs = x.shape();
    const auto& ys = y.shape();
    assert(xs.size()==2 && ys.size()==2 && xs[1]==ys[0]);
    size_t l = xs[0], m = xs[1], n = ys[1];

    vector<float> out(l * n);
    for (size_t i = 0; i < l; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < m; ++k) {
                sum += x.at(i,k) * y.at(k,j);
            }
            out[i * n + j] = sum;
        }
    }
    return Tensor(move(out), {l, n});
}

Tensor Tensor::apply(std::function<float(float)> func) const {
    vector<float> result_data;
    result_data.reserve(data.size());
    for (float val : data) {
        result_data.push_back(func(val));
    }
    return Tensor(move(result_data), shape_);
}

Tensor Tensor::relu() const {
    return apply([](float x) { return max(0.0f, x); });
}

Tensor Tensor::sigmoid() const {
    return apply([](float x) { return 1.0f / (1.0f + exp(-x)); });
}

Tensor Tensor::randn(vector<size_t> shape, float mean, float std) {
    size_t total_size = 1;
    for (auto dim : shape) total_size *= dim;
    
    static random_device rd;
    static mt19937 gen(rd());
    normal_distribution<float> dis(mean, std);
    
    vector<float> data(total_size);
    for (auto& val : data) {
        val = dis(gen);
    }
    return Tensor(move(data), shape);
}

// Fixed: Add Tensor:: scope and remove duplicate 'static'
Tensor Tensor::zeros(vector<size_t> shape) {
    size_t total_size = 1;
    for (auto dim : shape) total_size *= dim;
    return Tensor(vector<float>(total_size, 0.0f), shape);
}

// ---- NGL I chatgpt'd these 2 broadcasting thangs... will come back to understand a bit more deeply cause im lost in this broadcasting bs ---
vector<size_t> Tensor::broadcast_shape(const vector<size_t>& a,
                                            const vector<size_t>& b) {
    size_t na = a.size(), nb = b.size(), nr = max(na, nb);
    vector<size_t> result(nr);
    for (size_t i = 0; i < nr; ++i) {
        size_t da = (i < nr - na ? 1 : a[i - (nr - na)]);
        size_t db = (i < nr - nb ? 1 : b[i - (nr - nb)]);
        if      (da == db)         result[i] = da;
        else if (da == 1)          result[i] = db;
        else if (db == 1)          result[i] = da;
        else throw invalid_argument("Incompatible broadcast shapes");
    }
    return result;
}

Tensor Tensor::broadcast_to(const Tensor& t,
                            const vector<size_t>& target_shape) {
    auto src_shape = t.shape();
    size_t nd = target_shape.size();
    vector<size_t> tstrides(nd, 1);
    for (int i = (int)nd - 2; i >= 0; --i)
        tstrides[i] = tstrides[i+1] * target_shape[i+1];

    size_t total = 1;
    for (auto d : target_shape) total *= d;
    vector<float> data2(total);

    size_t ns = src_shape.size();
    for (size_t idx = 0; idx < total; ++idx) {
        vector<size_t> mid(nd);
        size_t tmp = idx;
        for (size_t i = 0; i < nd; ++i) {
            mid[i] = tmp / tstrides[i];
            tmp    = tmp % tstrides[i];
        }
        vector<size_t> src_idx(ns);
        size_t offset = nd - ns;
        for (size_t i = 0; i < ns; ++i) {
            src_idx[i] = (src_shape[i] == 1 ? 0 : mid[i + offset]);
        }
        data2[idx] = t.at(src_idx);
    }
    return Tensor(move(data2), target_shape);
}

Tensor Tensor::sum_across_axis(int axis) const{
    const auto& S = shape_;
    size_t N = S.size();
    assert(axis < N);

    vector<size_t> new_shape;
    new_shape.reserve(N-1);
    for (size_t i = 0; i < N; ++i) {
        if (i != axis) new_shape.push_back(S[i]);
    }
    auto new_strides = compute_strides(new_shape);

    size_t new_size = 1;
    for (auto d : new_shape) new_size *= d;
    vector<float> new_data(new_size, 0.0f);

    for (size_t flat = 0; flat < data.size(); ++flat) {
        size_t rem = flat;
        vector<size_t> idx(N);
        for (size_t i = 0; i < N; ++i) {
            idx[i] = rem / strides_[i];
            rem    = rem % strides_[i];
        }
        vector<size_t> ridx;
        ridx.reserve(N-1);
        for (size_t i = 0; i < N; ++i) {
            if (i != axis) ridx.push_back(idx[i]);
        }
        size_t new_flat = 0;
        for (size_t i = 0; i < ridx.size(); ++i) {
            new_flat += ridx[i] * new_strides[i];
        }
        new_data[new_flat] += data[flat];
    }

    return Tensor(move(new_data), new_shape);
}

Tensor Tensor::fit_gradient_shape(const Tensor& grad, const vector<size_t>& target_shape) {
    vector<size_t> gradient_shape = grad.shape();
    vector<int> sum_across_axes;
   
    int grad_size = gradient_shape.size();
    int target_size = target_shape.size();
    // assuming gradient shape is a lot bigger, we wanna basically collapse any extra dimensions assuming we broadcasted from target shape
    for (int i = 0; i < grad_size; i++) {
        sum_across_axes.push_back(i);
    }

    // sum across all of these dimensions
    for (int i = 0; i < target_size; i++) {
        int dim_i = grad_size-target_size+i;
        if (target_shape[i] == 1 && gradient_shape[dim_i] > 1) {
            sum_across_axes.push_back(dim_i);
        }
    }

    sort(sum_across_axes.rbegin(), sum_across_axes.rend());
    Tensor result = grad;
    for (int axis: sum_across_axes) {
        result = result.sum_across_axis(axis);
    }
    return result;
}

void Tensor::print() {
    if (shape_.size() == 2) {
        size_t rows = shape_[0], cols = shape_[1];
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                cout << at(i,j) << (j+1==cols ? "" : " ");
            }
            cout << "\n";
        }
    } else {
        for (size_t i = 0; i < data.size(); ++i) {
            cout << data[i] << (i+1==data.size() ? "" : " ");
        }
        cout << "\n";
    }
}
