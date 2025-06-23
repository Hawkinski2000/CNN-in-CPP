#include <iostream>
#include "optim.h"
#include "Engine.h"
#include "cuda_utils.cuh"
#include <cuda_runtime.h>
#include "Tensor.h"
using namespace std;


// Destructor for the Optimizer class
Optimizer::~Optimizer() = default;

// Reset the gradients of all parameters
void Optimizer::zero_grad() {
    for (Tensor* tensor : params) {
        if (tensor->device == "cuda") {
            cuda_fill(tensor->grad.get(), tensor->numel(), 0.0f);
        } else {
            fill(tensor->grad.get(), tensor->grad.get() + tensor->numel(), 0.0f);
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
    if (params[0]->device == "cuda") {
        cuda_step();
    }
    else {
        for (Tensor* tensor : params) {
            for (size_t i = 0; i < tensor->numel(); i++) {
                tensor->data.get()[i] -= lr * tensor->grad.get()[i];
            }
        }
    }

    Engine::clear_graph(Engine::get_root());
}
