#include "prime.hpp"
#include "loss.hpp"
#include "optim.hpp"
using namespace std;

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
    vector<float> xs = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    vector<float> ys = {0.0f, 2.0f, 4.0f, 6.0f, 8.0f};

    auto w = Value::create(0.0f, "w");    
    auto b = Value::create(0.0f, "b");

    AdamW optim_w(0.01f, w);
    AdamW optim_b(0.01f, b);
    ValuePtr epoch_loss;

    int n_epochs = 10000;

    for (int epoch = 0; epoch < n_epochs; ++epoch) {

        // zero‐grad
        w->set_grad(0.0f);
        b->set_grad(0.0f);
        epoch_loss = Value::create(0, "epoch");

        for (size_t i = 0; i < xs.size(); ++i) {
            auto x = xs[i];
            auto y_true = ys[i];

            auto x_val = Value::create(x, "x");
            auto y_val = Value::create(y_true, "y");

            auto lin = Value::add(Value::mult(w, x_val), b);

            MSELoss loss_fn(y_val, lin);
            auto loss_i = loss_fn.forward();

            epoch_loss = Value::add(epoch_loss, loss_i);
        
        }
        
        auto invN = Value::create(1.0f / xs.size(), "invN"); // technically this is a scalar multiplication so no need to support division
        epoch_loss = Value::mult(epoch_loss, invN);

        epoch_loss->backward();
        optim_w.step();
        optim_b.step();

        if (epoch % 50 == 0) {
            cout << "Epoch " << epoch
                 << " loss=" << epoch_loss->get_val()
                 << "  w=" << w->get_val()
                 << "  b=" << b->get_val()
                 << "\n";
        }

        if (epoch == n_epochs-1) {
             epoch_loss->print();
        }
    }

    cout << "\nTrained model: y ≈ "
         << w->get_val() << "·x + " << b->get_val() << "\n";

    vector<ValuePtr> topo;
    unordered_set<Value*> visited;
    epoch_loss->topo_sort(topo, visited);
    epoch_loss->visualize(topo, "/Users/devpatel/Downloads/development/amd/cpp/microml/viz/graph");
   
    return 0;

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

    return 0;


}
