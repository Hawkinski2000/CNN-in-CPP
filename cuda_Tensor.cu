#include <iostream>
#include "Tensor.h"


// Kernel for cuda_exp()
__global__ void exp_kernel(const float* __restrict__ input, float* output, size_t total_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        output[idx] = expf(input[idx]);
    }
}

// Function to return the exponential of all elements in a tensor that runs on the GPU
Tensor Tensor::cuda_exp() {
    Tensor result(dimensions, true);

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    exp_kernel<<<blocks, threads>>>(data.get(), result.data.get(), total_elements);
    cudaDeviceSynchronize();

    return result;
}

// Kernel for cuda_log()
__global__ void log_kernel(const float* __restrict__ input, float* output, size_t total_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        output[idx] = logf(input[idx]);
    }
}

// Function to return the natural logarithm of all elements in a tensor that runs on the GPU
Tensor Tensor::cuda_log() {
    Tensor result(dimensions, true);

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    log_kernel<<<blocks, threads>>>(data.get(), result.data.get(), total_elements);
    cudaDeviceSynchronize();

    return result;
}
