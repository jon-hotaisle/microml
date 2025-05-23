#include "prime.hpp"
#include "loss.hpp"
#include "nn.hpp"
#include "optim.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

int main() {
    // Generate simple random binary classification data
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::normal_distribution<float> dis(0.0f, 1.0f);
    
    // Create 100 samples with 4 features each
    size_t n_samples = 100;
    size_t n_features = 4;
    
    std::vector<std::vector<float>> X;
    std::vector<float> y;
    
    std::cout << "Generating " << n_samples << " samples with " << n_features << " features...\n";
    
    for (int i = 0; i < n_samples; ++i) {
        std::vector<float> sample(n_features);
        for (int j = 0; j < n_features; ++j) {
            sample[j] = dis(gen);
        }
        
        // Simple rule: if sum of first 2 features > sum of last 2 features, label = 1
        float sum_first = sample[0] + sample[1];
        float sum_last = sample[2] + sample[3];
        y.push_back(sum_first > sum_last ? 1.0f : 0.0f);
        
        X.push_back(sample);
    }
    
    // Print some sample data
    std::cout << "Sample data (first 5 examples):\n";
    for (int i = 0; i < 5; ++i) {
        std::cout << "X[" << i << "] = [";
        for (int j = 0; j < n_features; ++j) {
            std::cout << std::fixed << std::setprecision(2) << X[i][j];
            if (j < n_features - 1) std::cout << ", ";
        }
        std::cout << "] -> y = " << y[i] << "\n";
    }
    
    // Split into train/test (80/20)
    int train_size = 80;
    std::vector<std::vector<float>> X_train(X.begin(), X.begin() + train_size);
    std::vector<float> y_train(y.begin(), y.begin() + train_size);
    std::vector<std::vector<float>> X_test(X.begin() + train_size, X.end());
    std::vector<float> y_test(y.begin() + train_size, y.end());
    
    std::cout << "\nTraining on " << train_size << " samples, testing on " << (n_samples - train_size) << " samples\n";
    
    // === TRAINING WITH CROSS ENTROPY ===
    std::cout << "\n=== TRAINING WITH CROSS ENTROPY LOSS ===\n";
    
    MLP mlp_ce(n_features, 8, 4, 1);  // 4->8->4->1 architecture
    std::vector<AdamW> opts_ce;
    for (auto& p : mlp_ce.parameters()) {
        opts_ce.emplace_back(0.01f, p);  // Learning rate 0.01
    }
    
    // Training loop
    for (int epoch = 0; epoch < 200; ++epoch) {
        mlp_ce.zero_grad();
        float epoch_loss = 0.0f;
        
        for (int i = 0; i < train_size; ++i) {
            // Create input tensor
            Tensor x_tensor(X_train[i], {1, n_features});
            auto x = Value::create(x_tensor, "x");
            
            // Forward pass
            auto pred = mlp_ce.forward(x);
            
            // Create target tensor
            Tensor y_tensor({y_train[i]}, {1, 1});
            auto target = Value::create(y_tensor, "target");
            
            // Compute loss
            CrossEntropyLoss loss_fn(target, pred);
            auto loss = loss_fn.forward();
            epoch_loss += loss->get_val();
            
            // Backward pass
            loss->backward(false);
        }
        
        // Update parameters
        for (auto& opt : opts_ce) opt.step();
        
        if (epoch % 25 == 0) {
            std::cout << "Epoch " << std::setw(3) << epoch 
                      << " | Loss: " << std::fixed << std::setprecision(4) 
                      << (epoch_loss / train_size) << std::endl;
        }
    }
    
    // Test Cross Entropy model
    int correct_ce = 0;
    std::cout << "\nCross Entropy Model Predictions (first 10 test samples):\n";
    for (int i = 0; i < std::min(10, (int)X_test.size()); ++i) {
        Tensor x_tensor(X_test[i], {1, n_features});
        auto x = Value::create(x_tensor, "x");
        auto pred = mlp_ce.forward(x);
        
        float pred_val = pred->get_tensor().at(0, 0);
        int predicted = pred_val > 0.5f ? 1 : 0;
        int actual = (int)y_test[i];
        
        if (predicted == actual) correct_ce++;
        
        std::cout << "Sample " << i << ": pred=" << std::fixed << std::setprecision(3) 
                  << pred_val << " (" << predicted << "), actual=" << actual 
                  << (predicted == actual ? " ✓" : " ✗") << "\n";
    }
    
    // Calculate full test accuracy for CE
    for (int i = 10; i < (int)X_test.size(); ++i) {
        Tensor x_tensor(X_test[i], {1, n_features});
        auto x = Value::create(x_tensor, "x");
        auto pred = mlp_ce.forward(x);
        float pred_val = pred->get_tensor().at(0, 0);
        if ((pred_val > 0.5f ? 1 : 0) == (int)y_test[i]) correct_ce++;
    }
    
    std::cout << "Cross Entropy Test Accuracy: " 
              << std::fixed << std::setprecision(1) 
              << (100.0f * correct_ce / X_test.size()) << "%\n";
    
    // === TRAINING WITH MSE LOSS ===
    std::cout << "\n=== TRAINING WITH MSE LOSS ===\n";
    
    MLP mlp_mse(n_features, 8, 4, 1);  // Same architecture
    std::vector<AdamW> opts_mse;
    for (auto& p : mlp_mse.parameters()) {
        opts_mse.emplace_back(0.01f, p);
    }
    
    // Training loop
    for (int epoch = 0; epoch < 200; ++epoch) {
        mlp_mse.zero_grad();
        float epoch_loss = 0.0f;
        
        for (int i = 0; i < train_size; ++i) {
            // Create input tensor
            Tensor x_tensor(X_train[i], {1, n_features});
            auto x = Value::create(x_tensor, "x");
            
            // Forward pass
            auto pred = mlp_mse.forward(x);
            
            // Create target tensor
            Tensor y_tensor({y_train[i]}, {1, 1});
            auto target = Value::create(y_tensor, "target");
            
            // Compute MSE loss
            MSELoss loss_fn(target, pred);
            auto loss = loss_fn.forward();
            epoch_loss += loss->get_val();
            
            // Backward pass
            loss->backward(false);
        }
        
        // Update parameters
        for (auto& opt : opts_mse) opt.step();
        
        if (epoch % 25 == 0) {
            std::cout << "Epoch " << std::setw(3) << epoch 
                      << " | Loss: " << std::fixed << std::setprecision(4) 
                      << (epoch_loss / train_size) << std::endl;
        }
    }
    
    // Test MSE model
    int correct_mse = 0;
    std::cout << "\nMSE Model Predictions (first 10 test samples):\n";
    for (int i = 0; i < std::min(10, (int)X_test.size()); ++i) {
        Tensor x_tensor(X_test[i], {1, n_features});
        auto x = Value::create(x_tensor, "x");
        auto pred = mlp_mse.forward(x);
        
        float pred_val = pred->get_tensor().at(0, 0);
        int predicted = pred_val > 0.5f ? 1 : 0;
        int actual = (int)y_test[i];
        
        if (predicted == actual) correct_mse++;
        
        std::cout << "Sample " << i << ": pred=" << std::fixed << std::setprecision(3) 
                  << pred_val << " (" << predicted << "), actual=" << actual 
                  << (predicted == actual ? " ✓" : " ✗") << "\n";
    }
    
    // Calculate full test accuracy for MSE
    for (int i = 10; i < (int)X_test.size(); ++i) {
        Tensor x_tensor(X_test[i], {1, n_features});
        auto x = Value::create(x_tensor, "x");
        auto pred = mlp_mse.forward(x);
        float pred_val = pred->get_tensor().at(0, 0);
        if ((pred_val > 0.5f ? 1 : 0) == (int)y_test[i]) correct_mse++;
    }
    
    std::cout << "MSE Test Accuracy: " 
              << std::fixed << std::setprecision(1) 
              << (100.0f * correct_mse / X_test.size()) << "%\n";
    
    // Compare results
    std::cout << "\n=== COMPARISON ===\n";
    std::cout << "Cross Entropy Accuracy: " << (100.0f * correct_ce / X_test.size()) << "%\n";
    std::cout << "MSE Accuracy: " << (100.0f * correct_mse / X_test.size()) << "%\n";
    
    return 0;
}