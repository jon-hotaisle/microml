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

    static std::vector<size_t> broadcast_shape(const std::vector<size_t>& a,
                                               const std::vector<size_t>& b);
    static Tensor          broadcast_to(const Tensor& t,
                                        const std::vector<size_t>& target_shape);

    float& operator()(size_t i, size_t j) {
        return at({i,j});
    }
    const float& operator()(size_t i, size_t j) const {
        return at({i,j});
    }

    static Tensor matmul(const Tensor& x, const Tensor& y);
    void print();
};

// elementwise operators w. broadcasting
inline Tensor operator+(const Tensor& A, const Tensor& B) {
    auto out_shape = Tensor::broadcast_shape(A.shape(), B.shape());
    Tensor A2 = Tensor::broadcast_to(A, out_shape);
    Tensor B2 = Tensor::broadcast_to(B, out_shape);
    std::vector<float> out;
    out.reserve(A2.size());
    for (size_t i = 0; i < A2.size(); ++i) {
        out.push_back(A2.data[i] + B2.data[i]);
    }
    return Tensor(std::move(out), out_shape);
}

inline Tensor operator-(const Tensor& A, const Tensor& B) {
    auto out_shape = Tensor::broadcast_shape(A.shape(), B.shape());
    Tensor A2 = Tensor::broadcast_to(A, out_shape);
    Tensor B2 = Tensor::broadcast_to(B, out_shape);
    std::vector<float> out;
    out.reserve(A2.size());
    for (size_t i = 0; i < A2.size(); ++i) {
        out.push_back(A2.data[i] - B2.data[i]);
    }
    return Tensor(std::move(out), out_shape);
}

inline Tensor operator*(const Tensor& A, const Tensor& B) {
    auto out_shape = Tensor::broadcast_shape(A.shape(), B.shape());
    Tensor A2 = Tensor::broadcast_to(A, out_shape);
    Tensor B2 = Tensor::broadcast_to(B, out_shape);
    std::vector<float> out;
    out.reserve(A2.size());
    for (size_t i = 0; i < A2.size(); ++i) {
        out.push_back(A2.data[i] * B2.data[i]);
    }
    return Tensor(std::move(out), out_shape);
}
