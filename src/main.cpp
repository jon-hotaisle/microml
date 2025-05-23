#include "prime.hpp"
#include "loss.hpp"
#include "nn.hpp"
#include "optim.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <chrono>

using namespace std;

int main() {
    std::random_device rd;
    std::mt19937 gen(1513); 
    std::normal_distribution<float> dis(0.0f, 1.0f);
    
    size_t n_samples = 200;
    size_t n_features = 4;
    
    std::vector<std::vector<float>> X;
    std::vector<float> y;
    
    std::cout << "Generating " << n_samples << " samples with " << n_features << " features...\n";
    
    for (int i = 0; i < n_samples; ++i) {
        std::vector<float> sample(n_features);
        for (int j = 0; j < n_features; ++j) {
            sample[j] = dis(gen);
        }
        
        float sum_first = sample[0] + sample[1];
        float sum_last = sample[2] + sample[3];
        y.push_back(sum_first > sum_last ? 1.0f : 0.0f);
        
        X.push_back(sample);
    }
    
    std::cout << "Sample data (first 5 examples):\n";
    for (int i = 0; i < 5; ++i) {
        std::cout << "X[" << i << "] = [";
        for (int j = 0; j < n_features; ++j) {
            std::cout << std::fixed << std::setprecision(2) << X[i][j];
            if (j < n_features - 1) std::cout << ", ";
        }
        std::cout << "] -> y = " << y[i] << "\n";
    }
    
    int train_size = 160;
    std::vector<std::vector<float>> X_train(X.begin(), X.begin() + train_size);
    std::vector<float> y_train(y.begin(), y.begin() + train_size);
    std::vector<std::vector<float>> X_test(X.begin() + train_size, X.end());
    std::vector<float> y_test(y.begin() + train_size, y.end());
    
    std::cout << "\nTraining on " << train_size << " samples, testing on " << (n_samples - train_size) << " samples\n";
    
    std::cout << "\n-- Training with CE Loss --\n";

    MLP mlp_ce(n_features, 8, 4, 1);  // 4->8->4->1 architecture
    std::vector<AdamW> opts_ce; // set independent AdamW optimizer with shared learning parameters
    for (auto& p : mlp_ce.parameters()) {
        opts_ce.emplace_back(0.01f, p);
    }
    
    for (int epoch = 0; epoch < 200; ++epoch) {
        mlp_ce.zero_grad();
        float epoch_loss = 0.0f;
        int nan_count = 0;
        
        for (int i = 0; i < train_size; ++i) {
            Tensor x_tensor(X_train[i], {1, n_features});
            auto x = Value::create(x_tensor, "x");
            
            auto logits = mlp_ce.forward(x);
            
            auto pred = mlp_ce.sigmoid(logits);
            
            Tensor y_tensor({y_train[i]}, {1, 1});
            auto target = Value::create(y_tensor, "target");
            
            CrossEntropyLoss loss_fn(target, pred);  
            auto loss = loss_fn.forward();
            
            float loss_val = loss->get_val();
            
            float logits_val, pred_val, target_val;
            
            if (logits->get_tensor().is_scalar()) {
                logits_val = logits->get_val();
            } else {
                logits_val = logits->get_tensor().data[0]; // odd scalar transformation stuff, will need to debug
            }
            
            if (pred->get_tensor().is_scalar()) {
                pred_val = pred->get_val();
            } else {
                pred_val = pred->get_tensor().data[0];
            }
            
            if (target->get_tensor().is_scalar()) {
                target_val = target->get_val();
            } else {
                target_val = target->get_tensor().data[0];
            }
            
            if (std::isnan(loss_val) || std::isinf(loss_val)) {
                nan_count++;
                std::cout << "NaN/Inf detected at epoch " << epoch << ", sample " << i << std::endl;
                std::cout << "  Logits: " << logits_val << std::endl;
                std::cout << "  Pred: " << pred_val << std::endl;
                std::cout << "  Target: " << target_val << std::endl;
                continue;  
            }
            
            epoch_loss += loss_val;
            
            loss->backward(false);

            if (epoch == 199 && i == train_size-1) {
                std::cout << "Visualizing Cross Entropy Gradient Tree";
                vector<ValuePtr> topo;
                unordered_set<Value*> visited;
                loss->topo_sort(topo, visited);
                loss->visualize(topo, "/Users/devpatel/Downloads/development/amd/cpp/microml/viz/ce_test");
            }
        }
        
        if (nan_count > 0) {
            std::cout << "Warning: " << nan_count << " NaN/Inf losses in epoch " << epoch << std::endl;
        }
        
        for (auto& opt : opts_ce) opt.step();
        
        float avg_loss = epoch_loss / (train_size - nan_count);
        if (epoch % 25 == 0) {
            std::cout << "Epoch " << std::setw(3) << epoch 
                      << " | Loss: " << std::fixed << std::setprecision(4) 
                      << avg_loss << std::endl;
        }
        
    
    }
    
    int correct_ce = 0;
    std::cout << "\nCross Entropy Model Predictions (first 10 test samples):\n";
    for (int i = 0; i < std::min(10, (int)X_test.size()); ++i) {
        Tensor x_tensor(X_test[i], {1, n_features});
        auto x = Value::create(x_tensor, "x");
        auto logits = mlp_ce.forward(x);
        auto pred = mlp_ce.sigmoid(logits);  // sigmoid for CE since ReLU leads to NaNs for loss
        
        float pred_val;
        if (pred->get_tensor().is_scalar()) {
            pred_val = pred->get_val();
        } else {
            pred_val = pred->get_tensor().data[0]; 
        }
        int predicted = pred_val > 0.5f ? 1 : 0;
        int actual = (int)y_test[i];
        
        if (predicted == actual) correct_ce++;
        
        std::cout << "Sample " << i << ": pred=" << std::fixed << std::setprecision(3) 
                  << pred_val << " (" << predicted << "), actual=" << actual 
                  << (predicted == actual ? " ✓" : " ✗") << "\n";
    }
    
    for (int i = 10; i < (int)X_test.size(); ++i) {
        Tensor x_tensor(X_test[i], {1, n_features});
        auto x = Value::create(x_tensor, "x");
        auto logits = mlp_ce.forward(x);
        auto pred = mlp_ce.sigmoid(logits);
        
        float pred_val;
        if (pred->get_tensor().is_scalar()) {
            pred_val = pred->get_val();
        } else {
            pred_val = pred->get_tensor().data[0];
        }
        
        if ((pred_val > 0.5f ? 1 : 0) == (int)y_test[i]) correct_ce++;
    }
    
    std::cout << "Cross Entropy Test Accuracy: " 
              << std::fixed << std::setprecision(1) 
              << (100.0f * correct_ce / X_test.size()) << "%\n";
    
    std::cout << "\n=== TRAINING WITH MSE LOSS ===\n";
    
    MLP mlp_mse(n_features, 8, 4, 1); // same model architecture and approach
    std::vector<AdamW> opts_mse;
    for (auto& p : mlp_mse.parameters()) {
        opts_mse.emplace_back(0.01f, p);
    }
    
    for (int epoch = 0; epoch < 200; ++epoch) {
        mlp_mse.zero_grad();
        float epoch_loss = 0.0f;
        
        for (int i = 0; i < train_size; ++i) {
            Tensor x_tensor(X_train[i], {1, n_features});
            auto x = Value::create(x_tensor, "x");
            
            auto logits = mlp_mse.forward(x);
            auto pred = mlp_mse.sigmoid(logits);
            
            Tensor y_tensor({y_train[i]}, {1, 1});
            auto target = Value::create(y_tensor, "target");
            
            MSELoss loss_fn(target, pred);
            auto loss = loss_fn.forward();
            epoch_loss += loss->get_val();
            
            loss->backward(false);
            if (epoch == 199 && i == train_size-1) {
                std::cout << "Visualizing Mean Squared Error Gradient Tree";
                vector<ValuePtr> topo;
                unordered_set<Value*> visited;
                loss->topo_sort(topo, visited);
                loss->visualize(topo, "/Users/devpatel/Downloads/development/amd/cpp/microml/viz/mse_test");
            }
        }
        
        for (auto& opt : opts_mse) opt.step();
        
        if (epoch % 25 == 0) {
            std::cout << "Epoch " << std::setw(3) << epoch 
                      << " | Loss: " << std::fixed << std::setprecision(4) 
                      << (epoch_loss / train_size) << std::endl;
        }
    }
    
    int correct_mse = 0;
    std::cout << "\nMSE Model Predictions (first 10 test samples):\n";
    for (int i = 0; i < std::min(10, (int)X_test.size()); ++i) {
        Tensor x_tensor(X_test[i], {1, n_features});
        auto x = Value::create(x_tensor, "x");
        auto logits = mlp_mse.forward(x);
        auto pred = mlp_mse.sigmoid(logits);
        
        float pred_val;
        if (pred->get_tensor().is_scalar()) {
            pred_val = pred->get_val();
        } else {
            pred_val = pred->get_tensor().data[0];
        }
        int predicted = pred_val > 0.5f ? 1 : 0;
        int actual = (int)y_test[i];
        
        if (predicted == actual) correct_mse++;
        
        std::cout << "Sample " << i << ": pred=" << std::fixed << std::setprecision(3) 
                  << pred_val << " (" << predicted << "), actual=" << actual 
                  << (predicted == actual ? " ✓" : " ✗") << "\n";
    }
    
    for (int i = 10; i < (int)X_test.size(); ++i) {
        Tensor x_tensor(X_test[i], {1, n_features});
        auto x = Value::create(x_tensor, "x");
        auto logits = mlp_mse.forward(x);
        auto pred = mlp_mse.sigmoid(logits);
        
        float pred_val;
        if (pred->get_tensor().is_scalar()) {
            pred_val = pred->get_val();
        } else {
            pred_val = pred->get_tensor().data[0];
        }
        
        if ((pred_val > 0.5f ? 1 : 0) == (int)y_test[i]) correct_mse++;
    }
    
    std::cout << "MSE Test Accuracy: " 
              << std::fixed << std::setprecision(1) 
              << (100.0f * correct_mse / X_test.size()) << "%\n";
    
    std::cout << "\n=== COMPARISON ===\n";
    std::cout << "Cross Entropy Accuracy: " << (100.0f * correct_ce / X_test.size()) << "%\n";
    std::cout << "MSE Accuracy: " << (100.0f * correct_mse / X_test.size()) << "%\n";


    
    return 0;
}