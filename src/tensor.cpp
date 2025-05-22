#include "tensor.hpp"
#include <cassert>
#include <iostream>
#include <algorithm>   
#include <stdexcept> 
#include <cmath>

using namespace std;

std::vector<size_t> Tensor::compute_strides(const std::vector<size_t>& shape) {
    int n = (int)shape.size();
    std::vector<size_t> s(n, 1);
    for (int k = n - 2; k >= 0; --k) {
        s[k] = s[k+1] * shape[k+1];
    }
    return s;
}

Tensor::Tensor(std::vector<float> data_, std::vector<size_t> shape_)
  : shape_(shape_)                    
  , strides_(compute_strides(shape_)) 
  , data(std::move(data_))           
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

const std::vector<size_t>& Tensor::shape() const {
    return shape_;
}

size_t Tensor::offset(const std::vector<size_t>& idx) const {
    assert(idx.size() == shape_.size());
    size_t off = 0;
    for (size_t k = 0; k < idx.size(); ++k) {
        assert(idx[k] < shape_[k]);
        off += idx[k] * strides_[k];
    }
    return off;
}

float& Tensor::at(const std::vector<size_t>& idx) {
    return data[offset(idx)];
}

const float& Tensor::at(const std::vector<size_t>& idx) const {
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
    std::vector<float> new_data(data.size());
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            new_data[j*rows + i] = at(i,j);
        }
    }
    return Tensor(std::move(new_data), {cols, rows});
}

Tensor Tensor::matmul(const Tensor& x, const Tensor& y) {
    const auto& xs = x.shape();
    const auto& ys = y.shape();
    assert(xs.size()==2 && ys.size()==2 && xs[1]==ys[0]);
    size_t l = xs[0], m = xs[1], n = ys[1];

    std::vector<float> out(l * n);
    for (size_t i = 0; i < l; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < m; ++k) {
                sum += x.at(i,k) * y.at(k,j);
            }
            out[i * n + j] = sum;
        }
    }
    return Tensor(std::move(out), {l, n});
}

std::vector<size_t> Tensor::broadcast_shape(const std::vector<size_t>& a,
                                            const std::vector<size_t>& b) {
    size_t na = a.size(), nb = b.size(), nr = std::max(na, nb);
    std::vector<size_t> result(nr);
    for (size_t i = 0; i < nr; ++i) {
        size_t da = (i < nr - na ? 1 : a[i - (nr - na)]);
        size_t db = (i < nr - nb ? 1 : b[i - (nr - nb)]);
        if      (da == db)         result[i] = da;
        else if (da == 1)          result[i] = db;
        else if (db == 1)          result[i] = da;
        else throw std::invalid_argument("Incompatible broadcast shapes");
    }
    return result;
}

Tensor Tensor::broadcast_to(const Tensor& t,
                            const std::vector<size_t>& target_shape) {
    auto src_shape = t.shape();
    size_t nd = target_shape.size();
    std::vector<size_t> tstrides(nd, 1);
    for (int i = (int)nd - 2; i >= 0; --i)
        tstrides[i] = tstrides[i+1] * target_shape[i+1];

    size_t total = 1;
    for (auto d : target_shape) total *= d;
    std::vector<float> data2(total);

    size_t ns = src_shape.size();
    for (size_t idx = 0; idx < total; ++idx) {
        std::vector<size_t> mid(nd);
        size_t tmp = idx;
        for (size_t i = 0; i < nd; ++i) {
            mid[i] = tmp / tstrides[i];
            tmp    = tmp % tstrides[i];
        }
        std::vector<size_t> src_idx(ns);
        size_t offset = nd - ns;
        for (size_t i = 0; i < ns; ++i) {
            src_idx[i] = (src_shape[i] == 1 ? 0 : mid[i + offset]);
        }
        data2[idx] = t.at(src_idx);
    }
    return Tensor(std::move(data2), target_shape);
}

void Tensor::print() {
    if (shape_.size() == 2) {
        size_t rows = shape_[0], cols = shape_[1];
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                std::cout << at(i,j) << (j+1==cols ? "" : " ");
            }
            std::cout << "\n";
        }
    } else {
        for (size_t i = 0; i < data.size(); ++i) {
            std::cout << data[i] << (i+1==data.size() ? "" : " ");
        }
        std::cout << "\n";
    }
}
