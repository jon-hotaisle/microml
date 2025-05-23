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
    std::mt19937 gen(69420);
    
    std::normal_distribution<float> dis(0.0f, 1.0f);
    size_t n_samples = 600;
    size_t n_features = 4;
    
    std::vector<std::vector<float>> X_gt;
    std::vector<float> y_gt; 
    std::cout << "----------- GREATER THAN GATE -----------" << "\n";
    auto start_gt_ce = std::chrono::high_resolution_clock::now();

    std::cout << "Generating " << n_samples << " samples with " << n_features << " features...\n";
    
    for (int i = 0; i < n_samples; ++i) {
        std::vector<float> sample(n_features);
        for (int j = 0; j < n_features; ++j) {
            sample[j] = dis(gen);
        }
        
        float sum_first = sample[0] + sample[1];
        float sum_last = sample[2] + sample[3];
        y_gt.push_back(sum_first > sum_last ? 1.0f : 0.0f);
        
        X_gt.push_back(sample);
    }
    
    std::cout << "Sample data (first 5 examples):\n";
    for (int i = 0; i < 5; ++i) {
        std::cout << "X[" << i << "] = [";
        for (int j = 0; j < n_features; ++j) {
            std::cout << std::fixed << std::setprecision(2) << X_gt[i][j];
            if (j < n_features - 1) std::cout << ", ";
        }
        std::cout << "] -> y = " << y_gt[i] << "\n";
    }
    
    int train_size = 500;
    std::vector<std::vector<float>> X_train_gt(X_gt.begin(), X_gt.begin() + train_size);
    std::vector<float> y_train_gt(y_gt.begin(), y_gt.begin() + train_size);
    std::vector<std::vector<float>> X_test_gt(X_gt.begin() + train_size, X_gt.end());
    std::vector<float> y_test_gt(y_gt.begin() + train_size, y_gt.end());
    
    std::cout << "\nTraining on " << train_size << " samples, testing on " << (n_samples - train_size) << " samples\n";
    
    std::cout << "\n-- Training with CE Loss --\n";

    MLP mlp_ce_gt(n_features, 8, 4, 1);
    std::vector<AdamW> opts_ce_gt;
    for (auto& p : mlp_ce_gt.parameters()) {
        opts_ce_gt.emplace_back(0.01f, p);
    }
    
    for (int epoch = 0; epoch < 200; ++epoch) {
        mlp_ce_gt.zero_grad();
        float epoch_loss = 0.0f;
        int nan_count = 0;
        
        for (int i = 0; i < train_size; ++i) {
            Tensor x_tensor(X_train_gt[i], {1, n_features});
            auto x = Value::create(x_tensor, "x");
            
            auto logits = mlp_ce_gt.forward(x);
            
            auto pred = mlp_ce_gt.sigmoid(logits);
            
            Tensor y_tensor({y_train_gt[i]}, {1, 1});
            auto target = Value::create(y_tensor, "target");
            
            CrossEntropyLoss loss_fn(target, pred);  
            auto loss = loss_fn.forward();
            
            float loss_val = loss->get_val();
            
            float logits_val, pred_val, target_val;
            
            if (logits->get_tensor().is_scalar()) {
                logits_val = logits->get_val();
            } else {
                logits_val = logits->get_tensor().data[0];
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
        
        for (auto& opt : opts_ce_gt) opt.step();
        
        float avg_loss = epoch_loss / (train_size - nan_count);
        if (epoch % 25 == 0) {
            std::cout << "Epoch " << std::setw(3) << epoch 
                      << " | Loss: " << std::fixed << std::setprecision(4) 
                      << avg_loss << std::endl;
        }
    }

    auto end_gt_ce = std::chrono::high_resolution_clock::now();
    auto duration_gt_ce = std::chrono::duration_cast<std::chrono::milliseconds>(end_gt_ce - start_gt_ce);
    
    std::cout << "Greater Than Gate (CE Loss) Training Time: " << duration_gt_ce.count() << " ms\n";
    
    int correct_ce_gt = 0;
    std::cout << "\nCross Entropy Model Predictions (first 10 test samples):\n";
    for (int i = 0; i < std::min(10, (int)X_test_gt.size()); ++i) {
        Tensor x_tensor(X_test_gt[i], {1, n_features});
        auto x = Value::create(x_tensor, "x");
        auto logits = mlp_ce_gt.forward(x);
        auto pred = mlp_ce_gt.sigmoid(logits);
        
        float pred_val;
        if (pred->get_tensor().is_scalar()) {
            pred_val = pred->get_val();
        } else {
            pred_val = pred->get_tensor().data[0]; 
        }
        int predicted = pred_val > 0.5f ? 1 : 0;
        int actual = (int)y_test_gt[i];
        
        if (predicted == actual) correct_ce_gt++;
        
        std::cout << "Sample " << i << ": pred=" << std::fixed << std::setprecision(3) 
                  << pred_val << " (" << predicted << "), actual=" << actual 
                  << (predicted == actual ? " ✓" : " ✗") << "\n";
    }
    
    for (int i = 10; i < (int)X_test_gt.size(); ++i) {
        Tensor x_tensor(X_test_gt[i], {1, n_features});
        auto x = Value::create(x_tensor, "x");
        auto logits = mlp_ce_gt.forward(x);
        auto pred = mlp_ce_gt.sigmoid(logits);
        
        float pred_val;
        if (pred->get_tensor().is_scalar()) {
            pred_val = pred->get_val();
        } else {
            pred_val = pred->get_tensor().data[0];
        }
        
        if ((pred_val > 0.5f ? 1 : 0) == (int)y_test_gt[i]) correct_ce_gt++;
    }
    
    std::cout << "Cross Entropy Test Accuracy: " 
              << std::fixed << std::setprecision(1) 
              << (100.0f * correct_ce_gt / X_test_gt.size()) << "%\n";
    
    std::cout << "\n=== TRAINING WITH MSE LOSS ===\n";

    auto start_gt_mse = std::chrono::high_resolution_clock::now();
    
    MLP mlp_mse_gt(n_features, 8, 4, 1);
    std::vector<AdamW> opts_mse_gt;
    for (auto& p : mlp_mse_gt.parameters()) {
        opts_mse_gt.emplace_back(0.01f, p);
    }
    
    for (int epoch = 0; epoch < 200; ++epoch) {
        mlp_mse_gt.zero_grad();
        float epoch_loss = 0.0f;
        
        for (int i = 0; i < train_size; ++i) {
            Tensor x_tensor(X_train_gt[i], {1, n_features});
            auto x = Value::create(x_tensor, "x");
            
            auto logits = mlp_mse_gt.forward(x);
            auto pred = mlp_mse_gt.sigmoid(logits);
            
            Tensor y_tensor({y_train_gt[i]}, {1, 1});
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
        
        for (auto& opt : opts_mse_gt) opt.step();
        
        if (epoch % 25 == 0) {
            std::cout << "Epoch " << std::setw(3) << epoch 
                      << " | Loss: " << std::fixed << std::setprecision(4) 
                      << (epoch_loss / train_size) << std::endl;
        }
    }
    
    auto end_gt_mse = std::chrono::high_resolution_clock::now();
    auto duration_gt_mse = std::chrono::duration_cast<std::chrono::milliseconds>(end_gt_mse - start_gt_mse);
    
    std::cout << "Greater Than Gate (MSE Loss) Training Time: " << duration_gt_mse.count() << " ms\n";
    
    int correct_mse_gt = 0;
    std::cout << "\nMSE Model Predictions (first 10 test samples):\n";
    for (int i = 0; i < std::min(10, (int)X_test_gt.size()); ++i) {
        Tensor x_tensor(X_test_gt[i], {1, n_features});
        auto x = Value::create(x_tensor, "x");
        auto logits = mlp_mse_gt.forward(x);
        auto pred = mlp_mse_gt.sigmoid(logits);
        
        float pred_val;
        if (pred->get_tensor().is_scalar()) {
            pred_val = pred->get_val();
        } else {
            pred_val = pred->get_tensor().data[0];
        }
        int predicted = pred_val > 0.5f ? 1 : 0;
        int actual = (int)y_test_gt[i];
        
        if (predicted == actual) correct_mse_gt++;
        
        std::cout << "Sample " << i << ": pred=" << std::fixed << std::setprecision(3) 
                  << pred_val << " (" << predicted << "), actual=" << actual 
                  << (predicted == actual ? " ✓" : " ✗") << "\n";
    }
    
    for (int i = 10; i < (int)X_test_gt.size(); ++i) {
        Tensor x_tensor(X_test_gt[i], {1, n_features});
        auto x = Value::create(x_tensor, "x");
        auto logits = mlp_mse_gt.forward(x);
        auto pred = mlp_mse_gt.sigmoid(logits);
        
        float pred_val;
        if (pred->get_tensor().is_scalar()) {
            pred_val = pred->get_val();
        } else {
            pred_val = pred->get_tensor().data[0];
        }
        
        if ((pred_val > 0.5f ? 1 : 0) == (int)y_test_gt[i]) correct_mse_gt++;
    }
    
    std::cout << "MSE Test Accuracy: " 
              << std::fixed << std::setprecision(1) 
              << (100.0f * correct_mse_gt / X_test_gt.size()) << "%\n";
    
    std::cout << "\n=== COMPARISON ===\n";
    std::cout << "Cross Entropy Accuracy: " << (100.0f * correct_ce_gt / X_test_gt.size()) << "%\n";
    std::cout << "MSE Accuracy: " << (100.0f * correct_mse_gt / X_test_gt.size()) << "%\n";

    n_samples = 800;
    n_features = 2;
    
    std::vector<std::vector<float>> X_xor;
    std::vector<float> y_xor;
    
    std::cout << "\n----------- XOR GATE -----------" << "\n";
       
    auto start_xor_ce = std::chrono::high_resolution_clock::now();

    std::cout << "Generating " << n_samples << " samples with " << n_features << " features...\n";

    for (int i = 0; i < n_samples; ++i) {
        std::vector<float> sample(n_features);
        for (int j = 0; j < n_features; ++j) {
            sample[j] = dis(gen);
        }
        
        bool a = sample[0] > 0.0f;
        bool b = sample[1] > 0.0f;
        y_xor.push_back((a && !b) || (!a && b) ? 1.0f : 0.0f);
        
        X_xor.push_back(sample);
    }

    std::cout << "Sample data (first 8 examples):\n";
    for (int i = 0; i < 8; ++i) {
        std::cout << "X[" << i << "] = [";
        for (int j = 0; j < n_features; ++j) {
            std::cout << std::fixed << std::setprecision(2) << X_xor[i][j];
            if (j < n_features - 1) std::cout << ", ";
        }
        std::cout << "] -> y = " << y_xor[i] << "\n";
    }

    train_size = 600;
    std::vector<std::vector<float>> X_train_xor(X_xor.begin(), X_xor.begin() + train_size);
    std::vector<float> y_train_xor(y_xor.begin(), y_xor.begin() + train_size);
    std::vector<std::vector<float>> X_test_xor(X_xor.begin() + train_size, X_xor.end());
    std::vector<float> y_test_xor(y_xor.begin() + train_size, y_xor.end());

    std::cout << "\nTraining on " << train_size << " samples, testing on " << (n_samples - train_size) << " samples\n";

    std::cout << "\n-- Training with CE Loss --\n";

    MLP mlp_ce_xor(n_features, 6, 4, 1);
    std::vector<AdamW> opts_ce_xor;
    for (auto& p : mlp_ce_xor.parameters()) {
        opts_ce_xor.emplace_back(0.01f, p);
    }

    for (int epoch = 0; epoch < 300; ++epoch) {
        mlp_ce_xor.zero_grad();
        float epoch_loss = 0.0f;
        int nan_count = 0;
        
        for (int i = 0; i < train_size; ++i) {
            Tensor x_tensor(X_train_xor[i], {1, n_features});
            auto x = Value::create(x_tensor, "x");
            
            auto logits = mlp_ce_xor.forward(x);
            
            auto pred = mlp_ce_xor.sigmoid(logits);
            
            Tensor y_tensor({y_train_xor[i]}, {1, 1});
            auto target = Value::create(y_tensor, "target");
            
            CrossEntropyLoss loss_fn(target, pred);  
            auto loss = loss_fn.forward();
            
            float loss_val = loss->get_val();
            
            float logits_val, pred_val, target_val;
            
            if (logits->get_tensor().is_scalar()) {
                logits_val = logits->get_val();
            } else {
                logits_val = logits->get_tensor().data[0];
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

            if (epoch == 299 && i == train_size-1) {
                std::cout << "Visualizing Cross Entropy Gradient Tree";
                vector<ValuePtr> topo;
                unordered_set<Value*> visited;
                loss->topo_sort(topo, visited);
                loss->visualize(topo, "/Users/devpatel/Downloads/development/amd/cpp/microml/viz/xor_ce_test");
            }
        }
        
        if (nan_count > 0) {
            std::cout << "Warning: " << nan_count << " NaN/Inf losses in epoch " << epoch << std::endl;
        }
        
        for (auto& opt : opts_ce_xor) opt.step();
        
        float avg_loss = epoch_loss / (train_size - nan_count);
        if (epoch % 50 == 0) {
            std::cout << "Epoch " << std::setw(3) << epoch 
                    << " | Loss: " << std::fixed << std::setprecision(4) 
                    << avg_loss << std::endl;
        }
    }

    auto end_xor_ce = std::chrono::high_resolution_clock::now();
    auto duration_xor_ce = std::chrono::duration_cast<std::chrono::milliseconds>(end_xor_ce - start_xor_ce);
    
    std::cout << "XOR Gate (CE Loss) Training Time: " << duration_xor_ce.count() << " ms\n";
    

    int correct_ce_xor = 0;
    std::cout << "\nCross Entropy Model Predictions (first 12 test samples):\n";
    for (int i = 0; i < std::min(12, (int)X_test_xor.size()); ++i) {
        Tensor x_tensor(X_test_xor[i], {1, n_features});
        auto x = Value::create(x_tensor, "x");
        auto logits = mlp_ce_xor.forward(x);
        auto pred = mlp_ce_xor.sigmoid(logits);
        
        float pred_val;
        if (pred->get_tensor().is_scalar()) {
            pred_val = pred->get_val();
        } else {
            pred_val = pred->get_tensor().data[0]; 
        }
        int predicted = pred_val > 0.5f ? 1 : 0;
        int actual = (int)y_test_xor[i];
        
        if (predicted == actual) correct_ce_xor++;
        
        std::cout << "Sample " << i << ": pred=" << std::fixed << std::setprecision(3) 
                << pred_val << " (" << predicted << "), actual=" << actual 
                << (predicted == actual ? " ✓" : " ✗") << "\n";
    }

    for (int i = 12; i < (int)X_test_xor.size(); ++i) {
        Tensor x_tensor(X_test_xor[i], {1, n_features});
        auto x = Value::create(x_tensor, "x");
        auto logits = mlp_ce_xor.forward(x);
        auto pred = mlp_ce_xor.sigmoid(logits);
        
        float pred_val;
        if (pred->get_tensor().is_scalar()) {
            pred_val = pred->get_val();
        } else {
            pred_val = pred->get_tensor().data[0];
        }
        
        if ((pred_val > 0.5f ? 1 : 0) == (int)y_test_xor[i]) correct_ce_xor++;
    }

    std::cout << "Cross Entropy Test Accuracy: " 
            << std::fixed << std::setprecision(1) 
            << (100.0f * correct_ce_xor / X_test_xor.size()) << "%\n";

    std::cout << "\n=== TRAINING WITH MSE LOSS ===\n";
    auto start_xor_mse = std::chrono::high_resolution_clock::now();

    MLP mlp_mse_xor(n_features, 6, 4, 1);
    std::vector<AdamW> opts_mse_xor;
    for (auto& p : mlp_mse_xor.parameters()) {
        opts_mse_xor.emplace_back(0.01f, p);
    }

    for (int epoch = 0; epoch < 300; ++epoch) {
        mlp_mse_xor.zero_grad();
        float epoch_loss = 0.0f;
        
        for (int i = 0; i < train_size; ++i) {
            Tensor x_tensor(X_train_xor[i], {1, n_features});
            auto x = Value::create(x_tensor, "x");
            
            auto logits = mlp_mse_xor.forward(x);
            auto pred = mlp_mse_xor.sigmoid(logits);
            
            Tensor y_tensor({y_train_xor[i]}, {1, 1});
            auto target = Value::create(y_tensor, "target");
            
            MSELoss loss_fn(target, pred);
            auto loss = loss_fn.forward();
            epoch_loss += loss->get_val();
            
            loss->backward(false);
            if (epoch == 299 && i == train_size-1) {
                std::cout << "Visualizing Mean Squared Error Gradient Tree";
                vector<ValuePtr> topo;
                unordered_set<Value*> visited;
                loss->topo_sort(topo, visited);
                loss->visualize(topo, "/Users/devpatel/Downloads/development/amd/cpp/microml/viz/xor_mse_test");
            }
        }
        
        for (auto& opt : opts_mse_xor) opt.step();
        
        if (epoch % 50 == 0) {
            std::cout << "Epoch " << std::setw(3) << epoch 
                    << " | Loss: " << std::fixed << std::setprecision(4) 
                    << (epoch_loss / train_size) << std::endl;
        }
    }

    auto end_xor_mse = std::chrono::high_resolution_clock::now();
    auto duration_xor_mse = std::chrono::duration_cast<std::chrono::milliseconds>(end_xor_mse - start_xor_mse);
    
    std::cout << "XOR Gate (MSE Loss) Training Time: " << duration_xor_mse.count() << " ms\n";
    

    int correct_mse_xor = 0;
    std::cout << "\nMSE Model Predictions (first 12 test samples):\n";
    for (int i = 0; i < std::min(12, (int)X_test_xor.size()); ++i) {
        Tensor x_tensor(X_test_xor[i], {1, n_features});
        auto x = Value::create(x_tensor, "x");
        auto logits = mlp_mse_xor.forward(x);
        auto pred = mlp_mse_xor.sigmoid(logits);
        
        float pred_val;
        if (pred->get_tensor().is_scalar()) {
            pred_val = pred->get_val();
        } else {
            pred_val = pred->get_tensor().data[0];
        }
        int predicted = pred_val > 0.5f ? 1 : 0;
        int actual = (int)y_test_xor[i];
        
        if (predicted == actual) correct_mse_xor++;
        
        std::cout << "Sample " << i << ": pred=" << std::fixed << std::setprecision(3) 
                << pred_val << " (" << predicted << "), actual=" << actual 
                << (predicted == actual ? " ✔" : " 𝙭") << "\n";
    }

    for (int i = 12; i < (int)X_test_xor.size(); ++i) {
        Tensor x_tensor(X_test_xor[i], {1, n_features});
        auto x = Value::create(x_tensor, "x");
        auto logits = mlp_mse_xor.forward(x);
        auto pred = mlp_mse_xor.sigmoid(logits);
        
        float pred_val;
        if (pred->get_tensor().is_scalar()) {
            pred_val = pred->get_val();
        } else {
            pred_val = pred->get_tensor().data[0];
        }
        
        if ((pred_val > 0.5f ? 1 : 0) == (int)y_test_xor[i]) correct_mse_xor++;
    }

    std::cout << "MSE Test Accuracy: " 
            << std::fixed << std::setprecision(1) 
            << (100.0f * correct_mse_xor / X_test_xor.size()) << "%\n";

    std::cout << "\n=== XOR COMPARISON ===\n";
    std::cout << "Cross Entropy Accuracy: " << (100.0f * correct_ce_xor / X_test_xor.size()) << "%\n";
    std::cout << "MSE Accuracy: " << (100.0f * correct_mse_xor / X_test_xor.size()) << "%\n" << "%\n";

    std::cout << "\n=== TIMING SUMMARY ===\n";
    std::cout << "Greater Than Gate CE: " << duration_gt_ce.count() << " ms\n";
    std::cout << "Greater Than Gate MSE: " << duration_gt_mse.count() << " ms\n";
    std::cout << "XOR Gate CE: " << duration_xor_ce.count() << " ms\n";
    std::cout << "XOR Gate MSE: " << duration_xor_mse.count() << " ms\n";
    std::cout << "Total Training Time: " << (duration_gt_ce.count() + duration_gt_mse.count() + duration_xor_ce.count() + duration_xor_mse.count()) << " ms\n";

    return 0;
}