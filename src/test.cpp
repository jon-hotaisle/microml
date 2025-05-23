#include "prime.hpp"
#include "loss.hpp"
#include "nn.hpp"
#include "optim.hpp"
using namespace std;

bool almost_eq(float a, float b, float tol=1e-6f) {
    return std::fabs(a - b) < tol;
}

int test() {
    
    // basic arithmetic check to track trace
    // auto a = Value::create(1.0, "");
    // auto b = Value::create(2.0, "");

    // auto c = Value::add(a, b);
    // auto d = Value::sub(b, Value::mult(b, a));
    // d->print();
    // auto e = Value::mult(c, d);
    // e->print();

    // auto loss = new CrossEntropyLoss(d, a);
    // auto item = loss->forward();
    // item->print();

    // return 0;

    // example neural network forward pass
    // auto x = Value::create(1.0f, "x");   // Input feature
    // auto w = Value::create(0.5f, "w");   // Weight
    // auto b = Value::create(0.1f, "b");   // Bias
    // auto y = Value::create(1.0f, "y"); // Ground truth (binary classification)

    // auto z = Value::add(Value::mult(w, x), b);  // z = w*x + b

    // ReLU relu(z);
    // auto y_pred = relu.forward();

    // CrossEntropyLoss loss(y, y_pred);
    // auto final_loss = loss.forward();
    
    // cout << "Final Loss = " << final_loss->get_val() << "\n";
    // final_loss->backward();
    // cout << "\nComputation Graph:\n";
    // final_loss->print(true);

    // return 0;

    // 2 layer with residual connection
    // auto x      = Value::create(1.0f, "x");   // input feature
    // auto y_true = Value::create(1.0f, "y");   // target (binary)

    // auto w1 = Value::create(0.5f, "w1");
    // auto b1 = Value::create(0.1f, "b1");

    // auto w2 = Value::create(0.8f, "w2");
    // auto b2 = Value::create(0.2f, "b2");

    // auto z1 = Value::add(Value::mult(w1, x), b1);
    // ReLU relu1(z1);
    // auto h1 = relu1.forward();               
    // auto z2 = Value::add(Value::mult(w2, h1), b2);
    // auto res = Value::add(z2, h1);

    // ReLU relu2(res);
    // auto y_pred = relu2.forward();            

    // CrossEntropyLoss loss(y_true, y_pred);
    // auto final_loss = loss.forward();

    // std::cout << "Final Loss = " << final_loss->get_val() << "\n\n";

    // final_loss->backward();
    // std::cout << "Computation Graph:\n";
    // final_loss->print(true);

    // return 0;

    // learning y = 2x weights
    // vector<float> xs = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    // vector<float> ys = {0.0f, 2.0f, 4.0f, 6.0f, 8.0f};

    // auto w = Value::create(0.0f, "w");    
    // auto b = Value::create(0.0f, "b");

    // AdamW optim_w(0.01f, w);
    // AdamW optim_b(0.01f, b);
    
    // int n_epochs = 10000;

    // for (int epoch = 0; epoch < n_epochs; ++epoch) {
    //     float epoch_total_loss = 0.0f;
        
    //     w->set_grad(Tensor(0.0f));
    //     b->set_grad(Tensor(0.0f));
        
    //     for (size_t i = 0; i < xs.size(); ++i) {
    //         auto x_val = Value::create(xs[i], "x");
    //         auto y_val = Value::create(ys[i], "y");

    //         auto lin = Value::add(Value::mult(w, x_val), b);

    //         MSELoss loss_fn(y_val, lin);
    //         auto loss_i = loss_fn.forward();
            
    //         epoch_total_loss += loss_i->get_val();
            
    //         loss_i->backward(false);
            
    //     }
        
    //     w->set_grad(w->get_grad() * Tensor(1.0f / xs.size()));
    //     b->set_grad(b->get_grad() * Tensor(1.0f / xs.size()));
        
    //     optim_w.step();
    //     optim_b.step();
        
    //     float avg_loss = epoch_total_loss / xs.size();
        
    //     if (epoch % 50 == 0) {
    //         cout << "Epoch " << epoch
    //              << " loss=" << avg_loss
    //              << "  w=" << w->get_val()
    //              << "  b=" << b->get_val()
    //              << "\n";
    //     }
    // }

    // cout << "\nTrained model: y ≈ "
    //      << w->get_val() << "·x + " << b->get_val() << "\n";
    
    // return 0;
    // check operator overloading in native
    // auto a = Value::create(1.0f, "a");
    // auto b = Value::create(2.0f, "b");

    // auto c = a + b;            
    // auto d = b - b * a;        
    // auto e = c * d;           
    // auto f = pow(a, b);         
    // auto g = e / f;            

    // cout << "Forward results:\n";
    // cout << "  c = a + b = " << c->get_val() << "\n";
    // cout << "  d = b - b*a = " << d->get_val() << "\n";
    // cout << "  e = c*d     = " << e->get_val() << "\n";
    // cout << "  f = a^b     = " << f->get_val() << "\n";
    // cout << "  g = e/f     = " << g->get_val() << "\n\n";

    // auto y_true = Value::create(3.0f, "y_true");

    // MSELoss loss_fn(y_true, g); // order is ground truth to pred
    // auto loss = loss_fn.forward();

    // loss->backward();
    // cout << "After backprop:\n";
    // cout << "  dL/da = " << a->get_grad() << "\n";
    // cout << "  dL/db = " << b->get_grad() << "\n\n";
    // vector<ValuePtr> topo;
    // unordered_set<Value*> visited;
    // loss->topo_sort(topo, visited);
    // loss->visualize(topo, "/Users/devpatel/Downloads/development/amd/cpp/microml/viz/operator");


    // check matrix operations

    // Tensor A = Tensor({1, 2, 3, 4, 5, 6}, {2, 3});
    // Tensor B = Tensor({7, 8, 9, 10, 11, 12}, {2, 3});
    
    // B.print();
    // B = B.transpose();
    // B.print();
    
    // auto C = Tensor::matmul(A, B);
    // cout << "Matmul completed successfully" << endl;

    // C.print();
    // return 0;

    // some claude'd test code
    // cout << "=== Broadcasting Tests ===" << endl;
    
    // // Test 1: Same shape operations (should work without broadcasting)
    // cout << "\n1. Same shape operations:" << endl;
    // Tensor A = Tensor({1, 2, 3, 4}, {2, 2});
    // Tensor B = Tensor({5, 6, 7, 8}, {2, 2});
    
    // cout << "A:" << endl;
    // A.print();
    // cout << "B:" << endl;
    // B.print();
    
    // auto C = A + B;
    // cout << "A + B:" << endl;
    // C.print();
    
    // auto D = A * B;
    // cout << "A * B:" << endl;
    // D.print();
    
    // // Test 2: Scalar broadcasting
    // cout << "\n2. Scalar broadcasting:" << endl;
    // Tensor scalar = Tensor(10.0f);
    // Tensor matrix = Tensor({1, 2, 3, 4, 5, 6}, {2, 3});
    
    // cout << "Matrix:" << endl;
    // matrix.print();
    // cout << "Scalar: " << scalar.scalar_value() << endl;
    
    // auto result1 = matrix + scalar;
    // cout << "Matrix + Scalar:" << endl;
    // result1.print();
    
    // auto result2 = matrix * scalar;
    // cout << "Matrix * Scalar:" << endl;
    // result2.print();
    
    // // Test 3: Row vector broadcasting
    // cout << "\n3. Row vector broadcasting:" << endl;
    // Tensor mat = Tensor({1, 2, 3, 4, 5, 6}, {2, 3});
    // Tensor row_vec = Tensor({10, 20, 30}, {1, 3});
    
    // cout << "Matrix (2x3):" << endl;
    // mat.print();
    // cout << "Row vector (1x3):" << endl;
    // row_vec.print();
    
    // auto result3 = mat + row_vec;
    // cout << "Matrix + Row vector:" << endl;
    // result3.print();
    
    // // Test 4: Column vector broadcasting
    // cout << "\n4. Column vector broadcasting:" << endl;
    // Tensor col_vec = Tensor({100, 200}, {2, 1});
    
    // cout << "Matrix (2x3):" << endl;
    // mat.print();
    // cout << "Column vector (2x1):" << endl;
    // col_vec.print();
    
    // auto result4 = mat + col_vec;
    // cout << "Matrix + Column vector:" << endl;
    // result4.print();
    
    // // Test 5: Different dimension broadcasting
    // cout << "\n5. Different dimension broadcasting:" << endl;
    // Tensor vec = Tensor({1, 2, 3}, {3});  // 1D vector
    // Tensor mat2d = Tensor({10, 20, 30, 40, 50, 60}, {2, 3});  // 2D matrix
    
    // cout << "1D Vector (3,):" << endl;
    // vec.print();
    // cout << "2D Matrix (2x3):" << endl;
    // mat2d.print();
    
    // auto result5 = mat2d + vec;
    // cout << "2D Matrix + 1D Vector:" << endl;
    // result5.print();
    
    // // Test 6: Complex broadcasting example
    // cout << "\n6. Complex broadcasting:" << endl;
    // Tensor A_complex = Tensor({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {3, 4});
    // Tensor B_complex = Tensor({10, 20, 30, 40}, {1, 4});
    
    // cout << "Matrix A (3x4):" << endl;
    // A_complex.print();
    // cout << "Matrix B (1x4):" << endl;
    // B_complex.print();
    
    // auto result6 = A_complex * B_complex;
    // cout << "A * B (broadcasting):" << endl;
    // result6.print();
    
    // // Test 7: Subtraction broadcasting
    // cout << "\n7. Subtraction broadcasting:" << endl;
    // Tensor base = Tensor({10, 20, 30, 40, 50, 60}, {2, 3});
    // Tensor subtract_vec = Tensor({1, 2, 3}, {3});
    
    // cout << "Base matrix:" << endl;
    // base.print();
    // cout << "Subtract vector:" << endl;
    // subtract_vec.print();
    
    // auto result7 = base - subtract_vec;
    // cout << "Base - Vector:" << endl;
    // result7.print();
    
    // // Test 8: Error case (uncomment to test assertion)
    // cout << "\n8. Testing incompatible shapes (this should assert):" << endl;
    // cout << "Skipping incompatible shape test to avoid crash..." << endl;
    /*
    try {
        Tensor incompatible1 = Tensor({1, 2, 3}, {3});
        Tensor incompatible2 = Tensor({1, 2}, {2});
        auto should_fail = incompatible1 + incompatible2;  // Should assert
    } catch (...) {
        cout << "Caught expected error for incompatible shapes" << endl;
    }
    */
    
    // cout << "\n=== All broadcasting tests completed! ===" << endl;
    // return 0;

    // Generate simple random binary classification data
    // std::random_device rd;
    // std::mt19937 gen(42); // Fixed seed for reproducibility
    // std::normal_distribution<float> dis(0.0f, 1.0f);
    
    // // Create 100 samples with 4 features each
    // size_t n_samples = 200;
    // size_t n_features = 4;
    
    // std::vector<std::vector<float>> X;
    // std::vector<float> y;
    
    // std::cout << "Generating " << n_samples << " samples with " << n_features << " features...\n";
    
    // for (int i = 0; i < n_samples; ++i) {
    //     std::vector<float> sample(n_features);
    //     for (int j = 0; j < n_features; ++j) {
    //         sample[j] = dis(gen);
    //     }
        
    //     // Simple rule: if sum of first 2 features > sum of last 2 features, label = 1
    //     float sum_first = sample[0] + sample[1];
    //     float sum_last = sample[2] + sample[3];
    //     y.push_back(sum_first > sum_last ? 1.0f : 0.0f);
        
    //     X.push_back(sample);
    // }
    
    // std::cout << "Sample data (first 5 examples):\n";
    // for (int i = 0; i < 5; ++i) {
    //     std::cout << "X[" << i << "] = [";
    //     for (int j = 0; j < n_features; ++j) {
    //         std::cout << std::fixed << std::setprecision(2) << X[i][j];
    //         if (j < n_features - 1) std::cout << ", ";
    //     }
    //     std::cout << "] -> y = " << y[i] << "\n";
    // }
    
    // int train_size = 160;
    // std::vector<std::vector<float>> X_train(X.begin(), X.begin() + train_size);
    // std::vector<float> y_train(y.begin(), y.begin() + train_size);
    // std::vector<std::vector<float>> X_test(X.begin() + train_size, X.end());
    // std::vector<float> y_test(y.begin() + train_size, y.end());
    
    // std::cout << "\nTraining on " << train_size << " samples, testing on " << (n_samples - train_size) << " samples\n";
    
    // // === TRAINING WITH CROSS ENTROPY ===
    // std::cout << "\n=== TRAINING WITH CROSS ENTROPY LOSS ===\n";
    
    // MLP mlp_ce(n_features, 8, 4, 1);  // 4->8->4->1 architecture
    // std::vector<AdamW> opts_ce;
    // for (auto& p : mlp_ce.parameters()) {
    //     opts_ce.emplace_back(0.01f, p);  // Learning rate 0.01
    // }
    
    // // Training loop
    // for (int epoch = 0; epoch < 200; ++epoch) {
    //     mlp_ce.zero_grad();
    //     float epoch_loss = 0.0f;
        
    //     for (int i = 0; i < train_size; ++i) {
    //         // Create input tensor
    //         Tensor x_tensor(X_train[i], {1, n_features});
    //         auto x = Value::create(x_tensor, "x");
            
    //         // Forward pass
    //         auto pred = mlp_ce.forward(x);
            
    //         // Create target tensor
    //         Tensor y_tensor({y_train[i]}, {1, 1});
    //         auto target = Value::create(y_tensor, "target");
            
    //         // Compute loss
    //         CrossEntropyLoss loss_fn(target, pred);
    //         auto loss = loss_fn.forward();
    //         epoch_loss += loss->get_val();
            
    //         // Backward pass
    //         loss->backward(false);
    //     }
        
    //     // Update parameters
    //     for (auto& opt : opts_ce) opt.step();
        
    //     if (epoch % 25 == 0) {
    //         std::cout << "Epoch " << std::setw(3) << epoch 
    //                   << " | Loss: " << std::fixed << std::setprecision(4) 
    //                   << (epoch_loss / train_size) << std::endl;
    //     }
    // }
    
    // // Test Cross Entropy model
    // int correct_ce = 0;
    // std::cout << "\nCross Entropy Model Predictions (first 10 test samples):\n";
    // for (int i = 0; i < std::min(10, (int)X_test.size()); ++i) {
    //     Tensor x_tensor(X_test[i], {1, n_features});
    //     auto x = Value::create(x_tensor, "x");
    //     auto pred = mlp_ce.forward(x);
        
    //     float pred_val = pred->get_tensor().at(0, 0);
    //     int predicted = pred_val > 0.5f ? 1 : 0;
    //     int actual = (int)y_test[i];
        
    //     if (predicted == actual) correct_ce++;
        
    //     std::cout << "Sample " << i << ": pred=" << std::fixed << std::setprecision(3) 
    //               << pred_val << " (" << predicted << "), actual=" << actual 
    //               << (predicted == actual ? " ✓" : " ✗") << "\n";
    // }
    
    // // Calculate full test accuracy for CE
    // for (int i = 10; i < (int)X_test.size(); ++i) {
    //     Tensor x_tensor(X_test[i], {1, n_features});
    //     auto x = Value::create(x_tensor, "x");
    //     auto pred = mlp_ce.forward(x);
    //     float pred_val = pred->get_tensor().at(0, 0);
    //     if ((pred_val > 0.5f ? 1 : 0) == (int)y_test[i]) correct_ce++;
    // }
    
    // std::cout << "Cross Entropy Test Accuracy: " 
    //           << std::fixed << std::setprecision(1) 
    //           << (100.0f * correct_ce / X_test.size()) << "%\n";
    
    // // === TRAINING WITH MSE LOSS ===
    // std::cout << "\n=== TRAINING WITH MSE LOSS ===\n";
    
    // MLP mlp_mse(n_features, 8, 4, 1);  // Same architecture
    // std::vector<AdamW> opts_mse;
    // for (auto& p : mlp_mse.parameters()) {
    //     opts_mse.emplace_back(0.01f, p);
    // }
    
    // // Training loop
    // for (int epoch = 0; epoch < 200; ++epoch) {
    //     mlp_mse.zero_grad();
    //     float epoch_loss = 0.0f;
        
    //     for (int i = 0; i < train_size; ++i) {
    //         // Create input tensor
    //         Tensor x_tensor(X_train[i], {1, n_features});
    //         auto x = Value::create(x_tensor, "x");
            
    //         // Forward pass
    //         auto pred = mlp_mse.forward(x);
            
    //         // Create target tensor
    //         Tensor y_tensor({y_train[i]}, {1, 1});
    //         auto target = Value::create(y_tensor, "target");
            
    //         // Compute MSE loss
    //         MSELoss loss_fn(target, pred);
    //         auto loss = loss_fn.forward();
    //         epoch_loss += loss->get_val();
            
    //         // Backward pass
    //         loss->backward(false);
    //     }
        
    //     // Update parameters
    //     for (auto& opt : opts_mse) opt.step();
        
    //     if (epoch % 25 == 0) {
    //         std::cout << "Epoch " << std::setw(3) << epoch 
    //                   << " | Loss: " << std::fixed << std::setprecision(4) 
    //                   << (epoch_loss / train_size) << std::endl;
    //     }
    // }
    
    // // Test MSE model
    // int correct_mse = 0;
    // std::cout << "\nMSE Model Predictions (first 10 test samples):\n";
    // for (int i = 0; i < std::min(10, (int)X_test.size()); ++i) {
    //     Tensor x_tensor(X_test[i], {1, n_features});
    //     auto x = Value::create(x_tensor, "x");
    //     auto pred = mlp_mse.forward(x);
        
    //     float pred_val = pred->get_tensor().at(0, 0);
    //     int predicted = pred_val > 0.5f ? 1 : 0;
    //     int actual = (int)y_test[i];
        
    //     if (predicted == actual) correct_mse++;
        
    //     std::cout << "Sample " << i << ": pred=" << std::fixed << std::setprecision(3) 
    //               << pred_val << " (" << predicted << "), actual=" << actual 
    //               << (predicted == actual ? " ✓" : " ✗") << "\n";
    // }
    
    // // Calculate full test accuracy for MSE
    // for (int i = 10; i < (int)X_test.size(); ++i) {
    //     Tensor x_tensor(X_test[i], {1, n_features});
    //     auto x = Value::create(x_tensor, "x");
    //     auto pred = mlp_mse.forward(x);
    //     float pred_val = pred->get_tensor().at(0, 0);
    //     if ((pred_val > 0.5f ? 1 : 0) == (int)y_test[i]) correct_mse++;
    // }
    
    // std::cout << "MSE Test Accuracy: " 
    //           << std::fixed << std::setprecision(1) 
    //           << (100.0f * correct_mse / X_test.size()) << "%\n";
    
    // // Compare results
    // std::cout << "\n=== COMPARISON ===\n";
    // std::cout << "Cross Entropy Accuracy: " << (100.0f * correct_ce / X_test.size()) << "%\n";
    // std::cout << "MSE Accuracy: " << (100.0f * correct_mse / X_test.size()) << "%\n";
    
    // return 0;
}
