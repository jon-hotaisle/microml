### ML Compiler from Scratch in C++ [WORK IN PROGRESS]

- [Current] Micrograd Scalar Backprop and Layers
    - Operator Overloading
    - Toposort for Reduced Compile Time
    - End to End Train-Test Loop with AdamW
- [Next]
    - Vector and Matrix Support
    - Operator Fusion 
    - Computational Graph Optimization

Commands:
g++ -std=c++20 main.cpp nn.cpp optim.cpp prime.cpp tensor.cpp loss.cpp -o main
./main