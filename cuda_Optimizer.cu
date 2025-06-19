#include <iostream>
#include "optim.h"
#include "Tensor.h"


// Kernel for cuda_step()
__global__ void sgd_step_kernel(float* __restrict__ data, const float* __restrict__ grad, float lr, size_t total_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        data[idx] -= lr * grad[idx];
    }
}

// Function to perform a single optimization step to update parameters that runs on the GPU
void SGD::cuda_step() {
    for (Tensor* tensor : params) {
        int threads = 256;
        int blocks = (tensor->numel() + threads - 1) / threads;
        sgd_step_kernel<<<blocks, threads>>>(tensor->data.get(), tensor->grad.get(), lr, tensor->numel());
    }

    cudaDeviceSynchronize();
}
