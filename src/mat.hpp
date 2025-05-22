#pragma once

#include <vector>
#include <cstddef>
#include <utility>
#include <cassert>

class Tensor {
private:
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
public:
    std::vector<float> data;
    static std::vector<size_t> compute_strides(const std::vector<size_t>& shape);
    Tensor(std::vector<float> data_, std::vector<size_t> shape_);
    Tensor(float x);
    size_t size() const;
    const std::vector<size_t>& shape() const;
    size_t offset(const std::vector<size_t>& idx) const;
    float& at(const std::vector<size_t>& idx);
    const float& at(const std::vector<size_t>& idx) const;
    float& at(size_t i, size_t j);
    const float& at(size_t i, size_t j) const;
    bool is_scalar() const;
    float scalar_value() const;
    Tensor transpose() const;
};

// elementwise operators
inline Tensor operator+(const Tensor& A, const Tensor& B) {
    assert(A.shape() == B.shape());
    std::vector<float> out;
    out.reserve(A.size());
    for (size_t i = 0; i < A.size(); ++i) {
        out.push_back(A.data[i] + B.data[i]);
    }
    return Tensor(std::move(out), A.shape());
}

inline Tensor operator-(const Tensor& A, const Tensor& B) {
    assert(A.shape() == B.shape());
    std::vector<float> out;
    out.reserve(A.size());
    for (size_t i = 0; i < A.size(); ++i) {
        out.push_back(A.data[i] - B.data[i]);
    }
    return Tensor(std::move(out), A.shape());
}

inline Tensor operator*(const Tensor& A, const Tensor& B) {
    assert(A.shape() == B.shape());
    std::vector<float> out;
    out.reserve(A.size());
    for (size_t i = 0; i < A.size(); ++i) {
        out.push_back(A.data[i] * B.data[i]);
    }
    return Tensor(std::move(out), A.shape());
}
