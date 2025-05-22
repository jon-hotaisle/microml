#include "prime.hpp"
#include "loss.hpp"

int main() {
    
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
    auto x      = Value::create(1.0f, "x");   // input feature
    auto y_true = Value::create(1.0f, "y");   // target (binary)

    auto w1 = Value::create(0.5f, "w1");
    auto b1 = Value::create(0.1f, "b1");

    auto w2 = Value::create(0.8f, "w2");
    auto b2 = Value::create(0.2f, "b2");

    auto z1 = Value::add(Value::mult(w1, x), b1);
    ReLU relu1(z1);
    auto h1 = relu1.forward();               
    auto z2 = Value::add(Value::mult(w2, h1), b2);
    auto res = Value::add(z2, h1);

    ReLU relu2(res);
    auto y_pred = relu2.forward();            

    CrossEntropyLoss loss(y_true, y_pred);
    auto final_loss = loss.forward();

    std::cout << "Final Loss = " << final_loss->get_val() << "\n\n";

    final_loss->backward();
    std::cout << "Computation Graph:\n";
    final_loss->print(true);

    return 0;
}
