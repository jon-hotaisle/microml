#include "nn.hpp"
#include "prime.hpp"
#include "loss.hpp" 
#include "optim.hpp"
#include <random>

ValuePtr MLP::sigmoid(ValuePtr x) {

    Tensor input_tensor = x->get_tensor();
    Tensor output_tensor = input_tensor.sigmoid(); 
    
    auto node = Value::create(output_tensor, "sigmoid");
    node->set_prev({x});
    
    ValuePtr X = x, N = node;
    node->_backward = [X, N]() {
        Tensor node_grad = N->get_tensor_grad();
        Tensor output_tensor = N->get_tensor();  
        
        Tensor ones_tensor = Tensor::zeros(output_tensor.shape());
        for (size_t i = 0; i < ones_tensor.data.size(); ++i) {
            ones_tensor.data[i] = 1.0f;
        }
        
        Tensor sigmoid_grad = output_tensor * (ones_tensor - output_tensor);
        Tensor final_grad = node_grad * sigmoid_grad;
        X->add_grad(final_grad);
    };
    
    return node;
}

ValuePtr MLP::forward(ValuePtr x) {
    auto z1 = Value::add(Value::matmul(x, W1), b1);
    ReLU relu1(z1);
    auto a1 = relu1.forward();
    
    auto z2 = Value::add(Value::matmul(a1, W2), b2);
    ReLU relu2(z2);
    auto a2 = relu2.forward();
    
    auto z3 = Value::add(Value::matmul(a2, W3), b3);
    // return sigmoid(z3);
    return z3;
}


vector<ValuePtr> MLP::parameters() {
    return {W1, b1, W2, b2, W3, b3};
}

void MLP::zero_grad() {
    for (auto& param : parameters()) {
        param->set_tensor_grad(Tensor::zeros(param->get_tensor().shape()));
    }
}