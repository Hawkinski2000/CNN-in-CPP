#include <iostream>
#include "optim.h"
#include "Engine.h"
using namespace std;


// Destructor for the Optimizer class
Optimizer::~Optimizer() = default;

// Reset the gradients of all parameters
void Optimizer::zero_grad() {
    for (Tensor* tensor : params) {
        for (size_t i = 0; i < tensor->numel(); i++) {
            tensor->grad.get()[i] = 0;
        }
    }
}

// Constructor for the SGD class
SGD::SGD(vector<Tensor*> params, float lr) {
    this->params = params;
    this->lr = lr;
}

// Function to perform a single optimization step to update parameters
void SGD::step() {
    for (Tensor* tensor : params) {
        for (size_t i = 0; i < tensor->numel(); i++) {
            tensor->data.get()[i] -= lr * tensor->grad.get()[i];
        }
    }
    Engine::clear_graph(Engine::get_root());
}
