#include "mat.hpp"

std::vector<size_t> Tensor::compute_strides(const std::vector<size_t>& shape) {
    int n = (int)shape.size();
    std::vector<size_t> s(n, 1);
    for (int k = n - 2; k >= 0; --k) {
        s[k] = s[k+1] * shape[k+1];
    }
    return s;
}

Tensor::Tensor(std::vector<float> data_, std::vector<size_t> shape_)
  : data(std::move(data_))
  , shape_(std::move(shape_))
  , strides_(compute_strides(shape_))
{
    size_t expected = 1;
    for (auto d : shape_) expected *= d;
    assert(expected == data.size());
}

Tensor::Tensor(float x)
  : data({x})
  , shape_({})
  , strides_({})
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
