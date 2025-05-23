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

// static int read_be_int(std::ifstream &ifs) {
//     uint8_t bytes[4];
//     ifs.read(reinterpret_cast<char*>(bytes), 4);
//     return (bytes[0] << 24) | (bytes[1] << 16)
//          | (bytes[2] <<  8) |  bytes[3];
// }

// void load_mnist_images(const std::string &path,
//                        std::vector<std::vector<float>> &out_images,
//                        int &rows, int &cols)
// {
//     std::ifstream ifs(path, std::ios::binary);
//     if (!ifs) { 
//         std::cerr << "Cannot open " << path << "\n"; 
//         std::exit(1); 
//     }
    
//     int magic = read_be_int(ifs);
//     int nimgs = read_be_int(ifs);
//     rows = read_be_int(ifs);
//     cols = read_be_int(ifs);
    
//     std::cout << "Magic: " << magic << ", Images: " << nimgs 
//               << ", Rows: " << rows << ", Cols: " << cols << std::endl;
    
//     out_images.resize(nimgs, std::vector<float>(rows*cols));
//     for (int i = 0; i < nimgs; ++i) {
//         for (int j = 0; j < rows*cols; ++j) {
//             uint8_t pixel;
//             ifs.read(reinterpret_cast<char*>(&pixel), 1);
//             out_images[i][j] = pixel / 255.0f;
//         }
//     }
// }

// void load_mnist_labels(const std::string &path,
//                        std::vector<uint8_t> &out_labels)
// {
//     std::ifstream ifs(path, std::ios::binary);
//     if (!ifs) { 
//         std::cerr << "Cannot open " << path << "\n"; 
//         std::exit(1); 
//     }
    
//     int magic = read_be_int(ifs);
//     int nlab = read_be_int(ifs);
    
//     std::cout << "Labels - Magic: " << magic << ", Count: " << nlab << std::endl;
    
//     out_labels.resize(nlab);
//     for (int i = 0; i < nlab; ++i) {
//         uint8_t lbl;
//         ifs.read(reinterpret_cast<char*>(&lbl), 1);
//         out_labels[i] = lbl;
//     }
// }

// // Helper function to create one-hot encoded tensor

// // Convert label to one-hot
// Tensor create_one_hot(uint8_t label, size_t num_classes) {
//     vector<float> one_hot(num_classes, 0.0f);
//     one_hot[label] = 1.0f;
//     return Tensor(one_hot, {1, num_classes});
// }

// // Evaluate accuracy
// float evaluate_accuracy(MLP& mlp, const std::vector<std::vector<float>>& images, 
//                        const std::vector<uint8_t>& labels, size_t max_samples = 1000) {
//     size_t correct = 0;
//     size_t total = std::min(max_samples, images.size());
    
//     for (size_t i = 0; i < total; ++i) {
//         Tensor input_tensor(images[i], {1, images[i].size()});
//         auto x = Value::create(input_tensor, "x");
//         auto output = mlp.forward(x);
        
//         auto& logits = output->get_tensor().data;
//         int pred = std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
        
//         if (pred == labels[i]) {
//             correct++;
//         }
//     }
    
//     return static_cast<float>(correct) / total;
// }

// int main(int argc, char** argv) {
//     std::string base_path = "/Users/devpatel/Downloads/development/amd/cpp/microml/archive";
    
//     std::string train_imgs = base_path + "/train-images.idx3-ubyte";
//     std::string train_lbls = base_path + "/train-labels.idx1-ubyte";
//     std::string test_imgs  = base_path + "/t10k-images.idx3-ubyte";
//     std::string test_lbls  = base_path + "/t10k-labels.idx1-ubyte";
    
//     // Load data
//     std::vector<std::vector<float>> train_images, test_images;
//     std::vector<uint8_t> train_labels, test_labels;
//     int rows, cols;
    
//     std::cout << "Loading training data..." << std::endl;
//     load_mnist_images(train_imgs, train_images, rows, cols);
//     load_mnist_labels(train_lbls, train_labels);
    
//     std::cout << "Loading test data..." << std::endl;
//     load_mnist_images(test_imgs, test_images, rows, cols);
//     load_mnist_labels(test_lbls, test_labels);
    
//     std::cout << "Loaded " << train_images.size() << " training images and " 
//               << test_images.size() << " test images of size " << rows << "×" << cols << "\n";
    
//     // Create model (784 -> 128 -> 64 -> 10)
//     MLP mlp(rows * cols, 128, 64, 10);
    
//     // Create optimizers for each parameter
//     std::vector<AdamW> optimizers;
//     auto params = mlp.parameters();
//     float learning_rate = 0.001f;
    
//     for (auto& param : params) {
//         optimizers.emplace_back(learning_rate, param);
//     }
    
//     // Training parameters
//     int num_epochs = 5;  // Start with fewer epochs for testing
//     size_t batch_size = 32;
//     size_t num_batches = train_images.size() / batch_size;
    
//     std::cout << "Starting training with " << num_epochs << " epochs, batch size " 
//               << batch_size << ", " << num_batches << " batches per epoch" << std::endl;
    
//     // Initial accuracy
//     float initial_acc = evaluate_accuracy(mlp, test_images, test_labels, 1000);
//     std::cout << "Initial test accuracy: " << (initial_acc * 100.0f) << "%" << std::endl;
    
//     // Training loop
//     for (int epoch = 0; epoch < num_epochs; ++epoch) {
//         auto epoch_start = std::chrono::high_resolution_clock::now();
//         float total_loss = 0.0f;
        
//         // Shuffle training data
//         std::vector<size_t> indices(train_images.size());
//         std::iota(indices.begin(), indices.end(), 0);
//         std::random_device rd;
//         std::mt19937 g(rd());
//         std::shuffle(indices.begin(), indices.end(), g);
        
//         for (size_t batch = 0; batch < num_batches; ++batch) {
//             // Zero gradients
//             mlp.zero_grad();
            
//             float batch_loss = 0.0f;
            
//             // Process mini-batch
//             for (size_t i = 0; i < batch_size; ++i) {
//                 size_t idx = indices[batch * batch_size + i];
                
//                 // Create input tensor
//                 Tensor input_tensor(train_images[idx], {1, size_t(rows * cols)});
//                 auto x = Value::create(input_tensor, "x");
                
//                 // Create target one-hot
//                 Tensor target_tensor = create_one_hot(train_labels[idx], 10);
//                 auto target = Value::create(target_tensor, "target");
                
//                 // Forward pass
//                 auto logits = mlp.forward(x);
                
//                 // Compute loss using modified CrossEntropyLoss
//                 CrossEntropyLoss loss_fn(target, logits, 10);  // 10 classes for MNIST
//                 auto loss = loss_fn.forward();
                
//                 batch_loss += loss->get_val();
                
//                 // Backward pass
//                 loss->backward(false);
//             }
            
//             // Average gradients over batch
//             for (auto& param : params) {
//                 auto grad = param->get_grad();
//                 if (grad.is_scalar()) {
//                     param->set_grad(grad.scalar_value() / batch_size);
//                 } else {
//                     Tensor avg_grad = grad * Tensor(1.0f / batch_size);
//                     param->set_tensor_grad(avg_grad);
//                 }
//             }
            
//             // Update parameters
//             for (auto& optimizer : optimizers) {
//                 optimizer.step();
//             }
            
//             total_loss += batch_loss / batch_size;
            
//             // Print progress
//             if (batch % 2 == 0) {
//                 std::cout << "Epoch " << epoch + 1 << "/" << num_epochs 
//                           << ", Batch " << batch << "/" << num_batches 
//                           << ", Loss: " << (batch_loss / batch_size) << std::endl;
//             }
//         }
        
//         auto epoch_end = std::chrono::high_resolution_clock::now();
//         auto epoch_duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_end - epoch_start);
        
//         float test_acc = evaluate_accuracy(mlp, test_images, test_labels, 1000);
        
//         std::cout << "Epoch " << epoch + 1 << " completed in " << epoch_duration.count() 
//                   << "s, Avg Loss: " << (total_loss / num_batches) 
//                   << ", Test Accuracy: " << (test_acc * 100.0f) << "%" << std::endl;
//     }
    
//     std::cout << "\nTraining completed!" << std::endl;
//     float final_acc = evaluate_accuracy(mlp, test_images, test_labels, test_images.size());
//     std::cout << "Final test accuracy: " << (final_acc * 100.0f) << "%" << std::endl;
    
//     return 0;
// }



int main() {
    // Generate simple random binary classification data
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::normal_distribution<float> dis(0.0f, 1.0f);
    
    // Create 100 samples with 4 features each
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
        
        // Simple rule: if sum of first 2 features > sum of last 2 features, label = 1
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
        int nan_count = 0;
        
        for (int i = 0; i < train_size; ++i) {
            // Create input tensor
            Tensor x_tensor(X_train[i], {1, n_features});
            auto x = Value::create(x_tensor, "x");
            
            // Forward pass - get raw logits
            auto logits = mlp_ce.forward(x);
            
            // Apply sigmoid to get probabilities
            auto pred = mlp_ce.sigmoid(logits);
            
            // Create target tensor
            Tensor y_tensor({y_train[i]}, {1, 1});
            auto target = Value::create(y_tensor, "target");
            
            // Compute binary cross entropy loss (n_classes = 2 is default)
            CrossEntropyLoss loss_fn(target, pred);  // Binary CE
            auto loss = loss_fn.forward();
            
            float loss_val = loss->get_val();
            
            // Debug output for NaN cases
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
                continue;  // Skip this sample
            }
            
            epoch_loss += loss_val;
            
            // Backward pass
            loss->backward(false);
        }
        
        if (nan_count > 0) {
            std::cout << "Warning: " << nan_count << " NaN/Inf losses in epoch " << epoch << std::endl;
        }
        
        // Update parameters
        for (auto& opt : opts_ce) opt.step();
        
        float avg_loss = epoch_loss / (train_size - nan_count);
        if (epoch % 25 == 0) {
            std::cout << "Epoch " << std::setw(3) << epoch 
                      << " | Loss: " << std::fixed << std::setprecision(4) 
                      << avg_loss << std::endl;
        }
        
        // Early stopping if loss becomes NaN
        if (std::isnan(avg_loss)) {
            std::cout << "Training stopped due to NaN loss at epoch " << epoch << std::endl;
            break;
        }
    }
    
    // Test Cross Entropy model
    int correct_ce = 0;
    std::cout << "\nCross Entropy Model Predictions (first 10 test samples):\n";
    for (int i = 0; i < std::min(10, (int)X_test.size()); ++i) {
        Tensor x_tensor(X_test[i], {1, n_features});
        auto x = Value::create(x_tensor, "x");
        auto logits = mlp_ce.forward(x);
        auto pred = mlp_ce.sigmoid(logits);  // Apply sigmoid for prediction
        
        // Handle both scalar and tensor outputs
        float pred_val;
        if (pred->get_tensor().is_scalar()) {
            pred_val = pred->get_val();
        } else {
            pred_val = pred->get_tensor().data[0];  // Get first element if tensor
        }
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
            
            // Forward pass with sigmoid activation
            auto logits = mlp_mse.forward(x);
            auto pred = mlp_mse.sigmoid(logits);
            
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
    
    // Calculate full test accuracy for MSE
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
    
    // Compare results
    std::cout << "\n=== COMPARISON ===\n";
    std::cout << "Cross Entropy Accuracy: " << (100.0f * correct_ce / X_test.size()) << "%\n";
    std::cout << "MSE Accuracy: " << (100.0f * correct_mse / X_test.size()) << "%\n";
    
    return 0;
}