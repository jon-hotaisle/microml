#pragma once

#include <vector>
#include <cstddef>
#include <utility>
#include <cassert>
#include <functional>
using namespace std;


class Tensor {
    private:
        vector<size_t> shape_;
        vector<size_t> strides_;
    public:
        vector<float> data;
        Tensor() : data({0.0f}), strides_({}), shape_({}) {}
        static vector<size_t> compute_strides(const vector<size_t>& shape);
        Tensor(vector<float> data_, vector<size_t> shape_);
        Tensor(float x);
        size_t size() const;
        const vector<size_t>& shape() const;
        size_t offset(const vector<size_t>& idx) const;
        float& at(const vector<size_t>& idx);
        const float& at(const vector<size_t>& idx) const;
        float& at(size_t i, size_t j);
        const float& at(size_t i, size_t j) const;
        bool is_scalar() const;
        float scalar_value() const;
        Tensor transpose() const;

        static vector<size_t> broadcast_shape(const vector<size_t>& a,
                                                const vector<size_t>& b);
        static Tensor          broadcast_to(const Tensor& t,
                                            const vector<size_t>& target_shape);

        float& operator()(size_t i, size_t j) {
            return at({i,j});
        }
        const float& operator()(size_t i, size_t j) const {
            return at({i,j});
        }

        static Tensor matmul(const Tensor& x, const Tensor& y);
        static Tensor fit_gradient_shape(const Tensor& grad, const vector<size_t>& target_shape);
        Tensor sum_across_axis(int axis) const;
        void print();

        static Tensor zeros(vector<size_t> shape);
        static Tensor randn(vector<size_t> shape, float mean = 0.0f, float std = 1.0f);
        Tensor apply(std::function<float(float)> func) const;
        Tensor relu() const;
        Tensor sigmoid() const;
    };

// elementwise operators w. broadcasting
inline Tensor operator+(const Tensor& A, const Tensor& B) {
    auto out_shape = Tensor::broadcast_shape(A.shape(), B.shape());
    Tensor A2 = Tensor::broadcast_to(A, out_shape);
    Tensor B2 = Tensor::broadcast_to(B, out_shape);
    vector<float> out;
    out.reserve(A2.size());
    for (size_t i = 0; i < A2.size(); ++i) {
        out.push_back(A2.data[i] + B2.data[i]);
    }
    return Tensor(move(out), out_shape);
}

inline Tensor operator-(const Tensor& A, const Tensor& B) {
    auto out_shape = Tensor::broadcast_shape(A.shape(), B.shape());
    Tensor A2 = Tensor::broadcast_to(A, out_shape);
    Tensor B2 = Tensor::broadcast_to(B, out_shape);
    vector<float> out;
    out.reserve(A2.size());
    for (size_t i = 0; i < A2.size(); ++i) {
        out.push_back(A2.data[i] - B2.data[i]);
    }
    return Tensor(move(out), out_shape);
}

inline Tensor operator*(const Tensor& A, const Tensor& B) {
    auto out_shape = Tensor::broadcast_shape(A.shape(), B.shape());
    Tensor A2 = Tensor::broadcast_to(A, out_shape);
    Tensor B2 = Tensor::broadcast_to(B, out_shape);
    vector<float> out;
    out.reserve(A2.size());
    for (size_t i = 0; i < A2.size(); ++i) {
        out.push_back(A2.data[i] * B2.data[i]);
    }
    return Tensor(move(out), out_shape);
}
