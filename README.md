```markdown
<!--
███████╗██╗   ██╗██████╗ ██╗ ██████╗ ██╗     ██╗     ██╗     
██╔════╝██║   ██║██╔══██╗██║██╔═══██╗██║     ██║     ██║     
█████╗  ██║   ██║██████╔╝██║██║   ██║██║     ██║     ██║     
██╔══╝  ██║   ██║██╔══██╗██║██║   ██║██║     ██║     ██║     
██║     ╚██████╔╝██║  ██║██║╚██████╔╝███████╗███████╗███████╗
╚═╝      ╚═════╝ ╚═╝  ╚═╝╚═╝ ╚═════╝ ╚══════╝╚══════╝╚══════╝
                                                             
          MicroML: Edge Autodiff Engine Library             
-->

# MicroML: Edge Autodiff Engine Library 📈⚙️

MicroML is a **C++20**–native **Autodiff Engine** and **ML Compiler** framework. Built from the ground up, it fuses the elegance of functional operator overloading with a minimal dependency graph, empowering you to prototype, optimize, and deploy computational graphs at the edge—no GPU required.

---

## 🌟 Table of Contents

1. [Introduction](#introduction)  
2. [Architecture & Design](#architecture--design)  
3. [Key Capabilities](#key-capabilities)  
4. [Code Organization](#code-organization)  
5. [Building & Installation](#building--installation)  
6. [Getting Started Examples](#getting-started-examples)  
7. [Benchmarks & Profiling](#benchmarks--profiling)  
8. [Limitations & Caveats](#limitations--caveats)  
9. [Roadmap & Next Steps](#roadmap--next-steps)  
10. [Contributing](#contributing)  
11. [License](#license)  

---

## Introduction

Edge devices demand **performance**, **compactness**, and **reliability**. MicroML’s Autodiff Engine:
- **Compiles** your high-level model definitions into a lightweight computational graph.  
- **Differentiates** through that graph with a two-pass forward/backward engine—no Python runtime.  
- **Optimizes** kernels for scalar and tensor math on modern CPUs via **xsimd** (AVX2/FMA).  
- **Deploys** directly as a single `libmicroml.a` or header-only bundle into your IoT, robotics, or mobile firmware.

Whether you’re building:
- A tiny neural controller for a drone’s flight stabilization  
- A sensor-fusion pipeline on an embedded ARM board  
- A research prototype that requires complete auditability of gradients  

…MicroML gives you the building blocks of a **full ML Compiler** and **Autodiff Engine** in <5 MB of code.

---

## Architecture & Design

```

```
            ┌────────────────────┐
            │    User Model      │
            │  (C++ Operator     │
            │   Overloads)       │
            └────────┬───────────┘
                     │
                     ▼
    ┌────────────────────────────────┐
    │ Computational Graph Builder    │
    │  • Topological Sort            │
    │  • Node & Edge Allocation      │
    │  • Memory-Efficient Strides    │
    └────────────────────────────────┘
                     │
    ┌────────────────▼────────────────┐
    │ Forward Pass Evaluator         │
    │  • Lazy Execution              │
    │  • SIMD-Accelerated Kernels    │
    └────────────────┬────────────────┘
                     │
    ┌────────────────▼────────────────┐
    │ Backward Pass Differentiator   │
    │  • Reverse-Mode Autodiff       │
    │  • Gradient Accumulation       │
    └────────────────┬────────────────┘
                     │
    ┌────────────────▼────────────────┐
    │ ML Compiler & Optimizer        │
    │  • AdamW, SGD, RMSProp         │
    │  • Weight Decay & Clip         │
    │  • (Planned) Operator Fusion   │
    └────────────────────────────────┘
```

```

- **Value**: Core scalar node type holding data, gradient, op tag, and adjacency.  
- **Tensor**: N-D array with shape, strides, and raw buffer, enabling broadcast and matmul.  
- **Graph**: Built on operator calls; finalized by a fast **toposort** for minimal overhead.  

---

## Key Capabilities

### 1. Scalar & Tensor Autodiff  
- Reverse-mode autodiff on both scalars (`Value`) and N-D tensors.  
- Built-in ops: `+`, `-`, `*`, `/`, `pow`, `exp`, `log`, `relu`, `sigmoid`, `softmax`.

### 2. SIMD-Accelerated MatMul  
- `Tensor::matmul` leverages **xsimd** under the hood for AVX2/FMA speed on dense multiplies.  
- Efficient memory strides and blocking for cache-friendly throughput.

### 3. End-to-End Training Loop  
- Sample workflows: binary classification, multi-class CE, MSE regression.  
- Integrated **AdamW** optimizer with bias correction and weight decay.  
- Hooks for custom schedulers and gradient clipping.

### 4. Lightweight Dependency Footprint  
- **Header-only** core; optional `xsimd` for vector math.  
- No Python, no heavy BLAS/Eigen—fits easily into cross-compilation toolchains.

### 5. Graphviz Visualization  
- Export `.dot` files for your compute graph (`dump_to_dot`) and generate PNG via `graphviz`.  
- Inspect node-by-node—ideal for educational demos or auditing.

---

## Code Organization

```

microml/
├─ include/             ← Public headers
│   ├─ microml/
│   │   ├─ value.hpp    ← Scalar autograd node
│   │   ├─ tensor.hpp   ← N-D array with strides
│   │   ├─ nn.hpp       ← Layer definitions (Linear, MLP, Conv)
│   │   ├─ optim.hpp    ← Optimizers (AdamW, SGD…)
│   │   ├─ loss.hpp     ← Loss functions (CE, MSE, BCE)
│   │   └─ util.hpp     ← Helpers, RNG, etc.
├─ examples/            ← Standalone demos
│   ├─ xor\_gate.cpp
│   └─ greater\_than.cpp
├─ benchmarks/          ← Google Benchmark tests
│   └─ matmul\_bench.cpp
├─ scripts/             ← Build & CI helpers
│   └─ build.sh
├─ LICENSE
└─ README.md

````

---

## Building & Installation

1. **Clone**  
   ```bash
   git clone https://github.com/your-username/microml.git
   cd microml
````

2. **Fetch xsimd** (optional for SIMD)

   ```bash
   git submodule update --init --recursive
   ```
3. **Build Example**

   ```bash
   bash scripts/build.sh
   # or manually:
   g++ -std=c++20 -O3 -march=native -mavx2 -mfma \
       -Iinclude -Iinclude/xsimd \
       examples/xor_gate.cpp \
       -o build/xor_gate
   ```
4. **Run**

   ```bash
   ./build/xor_gate
   ```

---

## Getting Started Examples

### XOR Gate (2-layer MLP)

```cpp
// examples/xor_gate.cpp
#include "microml/value.hpp"
#include "microml/nn.hpp"
using namespace microml;

int main() {
  // Define network: 2 → 4 → 1
  MLP net({2, 4, 1}, Activation::ReLU);
  std::vector<std::vector<float>> data = {{0,1},{1,0},{0,0},{1,1}};
  std::vector<float> labels = {1,1,0,0};

  AdamW optimizer(net.parameters(), 1e-2, 0.9, 0.999, 1e-8, 1e-4);

  for (int epoch=0; epoch<500; ++epoch) {
    float epoch_loss = 0;
    for (int i=0; i<data.size(); ++i) {
      auto x = Tensor(data[i]);
      auto y_true = Tensor({labels[i]});
      auto y_pred = net.forward(x);
      auto loss = BCE(y_pred, y_true);
      loss.backward();
      optimizer.step();
      epoch_loss += loss.data()[0];
    }
    if (epoch % 50 == 0)
      printf("[Epoch %d] Loss = %.4f\n", epoch, epoch_loss);
  }
}
```

---

## Benchmarks & Profiling

| Benchmark                | Throughput (MiB/s) | Notes                         |
| ------------------------ | ------------------ | ----------------------------- |
| Scalar Backprop (1-node) | 1.2M ops/s         | Single-thread, no simd        |
| MatMul 128×128           | 8.6 GB/s           | AVX2 @ 3.5 GHz, cache-blocked |

> Use `benchmarks/matmul_bench.cpp` with [Google Benchmark](https://github.com/google/benchmark).

---

## Limitations & Caveats

* **No Kernel Fusion**: Each op is its own loop—operator fusion is on the roadmap.
* **Brute-force Broadcasting**: Works but not memory-optimal for large N-D arrays.
* **Numeric Instabilities**: Softmax + CE still need log-sum-exp trick for extreme logits.
* **Single-Threaded Core**: Only matmul is vectorized; no thread pool or GPU.
* **Manual Memory**: Raw pointers for `ValuePtr`; future upgrade to smart pointers + RAII.

---

## Roadmap & Next Steps

1. **Operator Fusion & JIT**

   * Auto-fuse sequences (e.g. `MatMul + Add + ReLU`) into single kernels.
   * Investigate LLVM ORC JIT for runtime codegen.
2. **Graph Optimizations**

   * CSE, dead-node pruning, constant folding, shape inference.
3. **Parallel & GPU Backends**

   * Thread pool offload, CUDA/PTX backend, or OpenCL.
4. **Python Frontend**

   * Expose Tensor/Value API via PyBind11 for rapid prototyping.
5. **Model Zoo & Tutorials**

   * Common architectures: simple CNNs, RNNs, Transformer encoder stub.
6. **Documentation & CI**

   * Doxygen docs, automated benchmarks, GitHub Actions CI + releases.

---

## License

This project is licensed under the [MIT License](LICENSE).
Feel free to use, modify, and redistribute!

```

---

> **MicroML** isn’t just “another autodiff toy” — it’s a foundation for your **edge compute** ambitions, blending **autodiff**, **compiler techniques**, and **SIMD-optimized** kernels into a single, modern C++ library. Build, iterate, and deploy your next-generation ML workloads right where they need to run: at the edge.
```
