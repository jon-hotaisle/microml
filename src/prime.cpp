#include "prime.hpp"
#include "loss.hpp"
#include <iostream>
#include <cmath>
#include <unordered_set>
#include <fstream> 
#include <vector>
#include <string>
#include <memory>
#include <sstream>

using namespace std;

Value::Value(float val, const string &op, size_t id): 
    data(val), grad(0.0f), op(op), id(id)
{}

Value::Value(const Tensor &t, const string &op, size_t id): 
    data(t), grad(Tensor(0.0f)), op(op), id(id) 
{}

ValuePtr Value::create(float val, const string &op) {
    return make_shared<Value>(val, op, currentID++);
}

ValuePtr Value::create(const Tensor &t, const string &op) {
    return make_shared<Value>(t, op, currentID++);
}
Value::~Value() {
    --Value::currentID;
}

void Value::print(bool verbose, int depth) {
    for (int i = 0; i < depth; ++i) {
        cout << "     ";
    }
    
    cout << "[data=" << data.scalar_value() 
              << ", grad=" << grad.scalar_value()
              << ", op=" << op 
              << ", id=" << id 
              << "]\n";

    if (verbose && !prev.empty()) {
        for (const auto& child : prev) {
            if (child) {
                child->print(verbose, depth + 1);
            }
        }
    }
}

float Value::get_val() const {
    return data.scalar_value();
}

void Value::set_val(float val) {
    data = Tensor(val);
}

void Value::set_prev(const vector<ValuePtr>& parents) {
    prev = parents;
}

void Value::set_grad(const Tensor &g) {
    grad = g;
}

void Value::set_grad(float g) {
    grad = Tensor(g);
}

const Tensor& Value::get_tensor() const {
    return data;
}
const Tensor& Value::get_tensor_grad() const {
    return grad;
}
void Value::set_tensor_grad(const Tensor& g) {
    grad = g;
}
void Value::add_tensor_grad(const Tensor& g) {
    grad = grad + g; 
}


void Value::add_grad(const Tensor &g) {
    grad = grad + g;
}

const Tensor& Value::get_grad() const {
    return grad;
}

string Value::get_op() const {
    return op;
}

ValuePtr Value::add(const ValuePtr& lhs, const ValuePtr& rhs) {
    Tensor out = lhs->data + rhs->data;
    auto node = Value::create(out, "+");
    node->set_prev({lhs, rhs});
    node->_backward = [lhs, rhs, node]() {
        Tensor g = node->get_grad();
        lhs->add_grad(Tensor::fit_gradient_shape(g, lhs->data.shape()));
        rhs->add_grad(Tensor::fit_gradient_shape(g, rhs->data.shape()));
    };
    return node;
}

ValuePtr Value::sub(const ValuePtr& lhs, const ValuePtr& rhs) {
    Tensor out = lhs->data - rhs->data;
    auto node = Value::create(out, "-");
    node->set_prev({lhs, rhs});
    node->_backward = [lhs, rhs, node]() {
        Tensor g = node->get_grad();
        lhs->add_grad(Tensor::fit_gradient_shape(g, lhs->data.shape()));
        rhs->add_grad(Tensor::fit_gradient_shape(-1*g, rhs->data.shape()));
    };
    return node;
}

ValuePtr Value::mult(const ValuePtr& lhs, const ValuePtr& rhs) {
    Tensor out = lhs->data * rhs->data;
    auto node = Value::create(out, "*");
    node->set_prev({lhs, rhs});
    node->_backward = [lhs, rhs, node]() {
        Tensor g = node->get_grad();
        lhs->add_grad(Tensor::fit_gradient_shape(rhs->data * g, lhs->data.shape()));
        rhs->add_grad(Tensor::fit_gradient_shape(lhs->data * g, rhs->data.shape()));
    };
    return node;
}

ValuePtr Value::exp(const ValuePtr& base, const ValuePtr& power) {
    float b = base->get_val();
    float p = power->get_val();
    float outv = pow(b, p);
    auto node = Value::create(outv, "^");
    node->set_prev({base, power});
    node->_backward = [base, power, node, outv]() {
        float g = node->get_grad().scalar_value();;
        float b = base->get_val();
        float p = power->get_val();
        base->add_grad(Tensor(g * p * pow(b, p - 1)));
        power->add_grad(Tensor(g * outv * log(b)));
    };
    return node;
}

ValuePtr Value::div(const ValuePtr& num, const ValuePtr& den) {
    auto inv = Value::exp(den, Value::create(-1.0f));
    return Value::mult(num, inv);
}

ValuePtr Value::divp(const ValuePtr& num, const ValuePtr& den) {
    float outv = num->get_val() / den->get_val();
    auto node = Value::create(outv, "/");
    node->set_prev({num, den});
    node->_backward = [num, den, node]() {
        float g = node->get_grad().scalar_value();
        num->add_grad(Tensor(g / den->get_val()));
        den->add_grad(Tensor(-g * num->get_val() / (den->get_val() * den->get_val())));
    };
    return node;
}

ValuePtr Value::matmul(const ValuePtr& A, const ValuePtr& B) {
    auto outTensor = Tensor::matmul(A->data, B->data);
    auto node = Value::create(outTensor, "matmul");
    node->set_prev({A,B});
    node->_backward = [A,B,node]() {
        Tensor g = node->get_grad();
        A->add_grad(Tensor::matmul(g, B->data.transpose())); // dA*B/d_ flips order, transpose for shape
        B->add_grad(Tensor::matmul(A->data.transpose(), g));
    };
    return node;
}

void Value::backward(bool retain_graph = false) {
    vector<ValuePtr> topo;
    unordered_set<Value*> visited;
    topo_sort(topo, visited);

    if (!retain_graph || (grad.is_scalar() && grad.scalar_value() == 0.0f)) {
        grad = Tensor({1.0f}, {1, 1});;
    }

    for (auto next = topo.rbegin(); next != topo.rend(); ++next) {
        auto& node = *next;
        if (node->_backward) {
            node->_backward();
        }
    }
}

void Value::topo_sort(vector<ValuePtr>& topo, unordered_set<Value*>& visited) {
    if (visited.count(this)) {
        return;
    }
    visited.insert(this);
    for (auto& p : prev) {
        p->topo_sort(topo, visited);
    }
    topo.push_back(shared_from_this());
}

void Value::set_tensor(const Tensor& t) {
    data = t;
}

void Value::dump_to_dot(const vector<ValuePtr>& topo, const string& filename) {
    // auto tensor_to_string = [](const Tensor& t) {
    //     std::ostringstream oss;
    //     oss << "[";
    //     for (size_t i = 0; i < t.data.size(); ++i) {
    //         oss << t.data[i];
    //         if (i + 1 < t.data.size()) oss << ",";
    //     }
    //     oss << "]";
    //     return oss.str();
    // };


    // ofstream out(filename);
    // out << "digraph ComputationalGraph {\n";
    // for (auto& node : topo) {
    //     const Tensor& val_t  = node->get_tensor();
    //     const Tensor& grad_t = node->get_tensor_grad();
    //     std::string val_str  = tensor_to_string(val_t);
    //     std::string grad_str = tensor_to_string(grad_t);

    //     out << "  node" << node->id
    //         << " [label=\"" << node->op
    //         << "\\nval="  << val_str
    //         << "\\ngrad=" << grad_str << "\"];\n";

    //     for (auto& parent : node->prev) {
    //         out << "  node" << parent->id
    //             << " -> node" << node->id << ";\n";
    //     }
    // }
    // out << "}\n";
    // out.close();

    // AI-Generated Prettier Version of my Prior Graph Code 
    auto tensor_to_string = [](const Tensor& t, int max_elements = 3) {
        std::ostringstream oss;
        if (t.is_scalar()) {
            oss << std::fixed << std::setprecision(3) << t.scalar_value();
        } else {
            oss << "[";
            size_t elements_to_show = std::min((size_t)max_elements, t.data.size());
            for (size_t i = 0; i < elements_to_show; ++i) {
                oss << std::fixed << std::setprecision(3) << t.data[i];
                if (i + 1 < elements_to_show) oss << ", ";
            }
            if (t.data.size() > max_elements) {
                oss << "...";
            }
            oss << "]";
        }
        return oss.str();
    };

    auto get_node_color = [](const string& op) {
        if (op == "matmul") return "lightblue";
        if (op == "+" || op == "-") return "lightgreen";
        if (op == "*" || op == "/") return "lightyellow";
        if (op == "relu") return "lightcoral";
        if (op == "sigmoid") return "lightpink";
        if (op == "bce" || op == "cross_entropy") return "orange";
        if (op == "mse") return "lightgray";
        if (op.empty() || op.find("W") != string::npos || op.find("b") != string::npos) return "lightsteelblue";
        return "white";
    };

    auto get_node_shape = [](const string& op) {
        if (op == "bce" || op == "cross_entropy" || op == "mse") return "diamond";
        if (op == "relu" || op == "sigmoid") return "ellipse";
        if (op == "matmul") return "box";
        if (op.empty() || op.find("W") != string::npos || op.find("b") != string::npos) return "circle";
        return "box";
    };

    auto escape_string = [](const string& str) {
        string result = str;
        size_t pos = 0;
        while ((pos = result.find("\"", pos)) != string::npos) {
            result.replace(pos, 1, "\\\"");
            pos += 2;
        }
        return result;
    };

    ofstream out(filename);
    
    out << "digraph ComputationalGraph {\n";
    out << "  // Graph styling\n";
    out << "  rankdir=TB;\n"; 
    out << "  bgcolor=\"white\";\n";
    out << "  node [fontname=\"Arial\", fontsize=10, margin=0.1];\n";
    out << "  edge [fontname=\"Arial\", fontsize=8, color=\"#333333\"];\n";
    out << "  \n";
    
    out << "  // Parameter nodes\n";
    for (auto& node : topo) {
        if (node->op.empty() || node->op.find("W") != string::npos || 
            node->op.find("b") != string::npos || node->op == "x" || node->op == "target") {
            
            string val_str = tensor_to_string(node->get_tensor());
            string grad_str = tensor_to_string(node->get_tensor_grad());
            string color = get_node_color(node->op);
            string shape = get_node_shape(node->op);
            
            out << "  node" << node->id
                << " [label=\"" << escape_string(node->op)
                << "\\nval=" << escape_string(val_str)
                << "\\ngrad=" << escape_string(grad_str) 
                << "\", fillcolor=\"" << color 
                << "\", style=\"filled\", shape=\"" << shape << "\"];\n";
        }
    }
    
    out << "  \n  // Operation nodes\n";
    for (auto& node : topo) {
        if (!(node->op.empty() || node->op.find("W") != string::npos || 
              node->op.find("b") != string::npos || node->op == "x" || node->op == "target")) {
            
            string val_str = tensor_to_string(node->get_tensor());
            string grad_str = tensor_to_string(node->get_tensor_grad());
            string color = get_node_color(node->op);
            string shape = get_node_shape(node->op);
            
            string label;
            if (node->op == "bce" || node->op == "cross_entropy" || node->op == "mse") {
                label = "LOSS\\n" + escape_string(node->op) + "\\n" + escape_string(val_str);
            } else {
                label = escape_string(node->op) + "\\nval=" + escape_string(val_str) + 
                       "\\ngrad=" + escape_string(grad_str);
            }
            
            out << "  node" << node->id
                << " [label=\"" << label
                << "\", fillcolor=\"" << color 
                << "\", style=\"filled\", shape=\"" << shape << "\"];\n";
        }
    }
    
    out << "  \n  // Edges\n";
    for (auto& node : topo) {
        for (auto& parent : node->prev) {
            string edge_style = "solid";
            string edge_color = "#333333";
            
            if (node->op == "matmul") {
                edge_color = "#0066CC";
                edge_style = "bold";
            } else if (node->op == "bce" || node->op == "cross_entropy" || node->op == "mse") {
                edge_color = "#CC3300";
                edge_style = "bold";
            } else if (node->op == "relu" || node->op == "sigmoid") {
                edge_color = "#006600";
            }
            
            out << "  node" << parent->id
                << " -> node" << node->id 
                << " [color=\"" << edge_color << "\", style=\"" << edge_style << "\"];\n";
        }
    }
    
    out << "  \n  // Legend\n";
    out << "  subgraph cluster_legend {\n";
    out << "    label=\"Legend\";\n";
    out << "    style=\"filled\";\n";
    out << "    fillcolor=\"#f0f0f0\";\n";
    out << "    fontsize=8;\n";
    out << "    \n";
    out << "    legend_param [label=\"Parameters\", fillcolor=\"lightsteelblue\", style=\"filled\", shape=\"circle\"];\n";
    out << "    legend_matmul [label=\"Matrix Ops\", fillcolor=\"lightblue\", style=\"filled\", shape=\"box\"];\n";
    out << "    legend_activation [label=\"Activations\", fillcolor=\"lightcoral\", style=\"filled\", shape=\"ellipse\"];\n";
    out << "    legend_loss [label=\"Loss\", fillcolor=\"orange\", style=\"filled\", shape=\"diamond\"];\n";
    out << "    \n";
    out << "    legend_param -> legend_matmul -> legend_activation -> legend_loss [style=\"invis\"];\n";
    out << "  }\n";
    
    out << "}\n";
    out.close();

}

void Value::visualize(const vector<ValuePtr>& topo, const string& base_path) {
    string dotfile = base_path + ".dot";
    string pngfile = base_path + ".png";
    dump_to_dot(topo, dotfile);
    string cmd = "dot -Tpng " + dotfile + " -o " + pngfile;
    cout << "Running: " << cmd << "\n";
    int ret = system(cmd.c_str());
    if (ret != 0) {
        cerr << "Error: dot command failed with code " << ret << "\n";
    } else {
        cout << "Graph written to " << pngfile << "\n";
    }
}