#include "prime.hpp"
#include "loss.hpp" 
#include "optim.hpp"
#include <random>

class TensorMLP {
    private:
        ValuePtr W1, b1;  
        ValuePtr W2, b2; 
        ValuePtr W3, b3;  
    public:
        TensorMLP(size_t input_size, size_t hidden1_size, size_t hidden2_size, size_t output_size) {
            float limit1 = sqrt(6.0f / (input_size + hidden1_size));
            float limit2 = sqrt(6.0f / (hidden1_size + hidden2_size));
            float limit3 = sqrt(6.0f / (hidden2_size + output_size));
            
            W1 = Value::create(Tensor::randn({input_size, hidden1_size}, 0.0f, limit1), "W1");
            b1 = Value::create(Tensor::zeros({1, hidden1_size}), "b1");  // Changed from {hidden1_size} to {1, hidden1_size}

            W2 = Value::create(Tensor::randn({hidden1_size, hidden2_size}, 0.0f, limit2), "W2");
            b2 = Value::create(Tensor::zeros({1, hidden2_size}), "b2");  // Changed from {hidden2_size} to {1, hidden2_size}
            
            W3 = Value::create(Tensor::randn({hidden2_size, output_size}, 0.0f, limit3), "W3");
            b3 = Value::create(Tensor::zeros({1, output_size}), "b3");   // Changed from {output_size} to {1, output_size}
        }


        ValuePtr forward(ValuePtr x);
        std::vector<ValuePtr> parameters();
        void zero_grad();
        ValuePtr sigmoid(ValuePtr x);
};