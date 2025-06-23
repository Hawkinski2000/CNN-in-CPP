#include <iostream>
// #include <curand.h>
#include <curand_kernel.h>
#include "Tensor.h"


// Kernel for cuda_rand()
__global__ void rand_kernel(float* data, size_t total_elements, float lower, float upper, uint64_t seed) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) {
        return;
    }

    curandState state;
    curand_init(seed, idx, 0, &state);

    float rand_val = curand_uniform(&state);
    data[idx] = lower + (upper - lower) * rand_val;
}

// Function to create a tensor of random values from a specified shape that runs on the GPU
Tensor Tensor::cuda_rand(initializer_list<size_t> dims, size_t in_features) {
    Tensor tensor(dims, true);
    if (in_features == 0) {
        in_features = tensor.dimensions[0];
    }

    float limit = sqrtf(1.0f / in_features);
    float lower = -limit;
    float upper = limit;

    uint64_t seed = static_cast<uint64_t>(time(nullptr));

    int threads = 1024;
    int blocks = (tensor.total_elements + threads - 1) / threads;
    rand_kernel<<<blocks, threads>>>(tensor.data.get(), tensor.total_elements, lower, upper, seed);
    
    

    return tensor;
}

// ---------------------------------------------------------------------------

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

    int threads = 1024;
    int blocks = (total_elements + threads - 1) / threads;
    exp_kernel<<<blocks, threads>>>(data.get(), result.data.get(), total_elements);
    

    return result;
}

// ---------------------------------------------------------------------------

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

    int threads = 1024;
    int blocks = (total_elements + threads - 1) / threads;
    log_kernel<<<blocks, threads>>>(data.get(), result.data.get(), total_elements);
    

    return result;
}

// ---------------------------------------------------------------------------

// Kernel for cuda_eq()
__global__ void cuda_eq_kernel(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C,
                               size_t total_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        if (A[idx] == B[idx]) {
            C[idx] = 1.0f;
        }
        else {
            C[idx] = 0.0f;
        }
    }
}

// Function for element-wise equality between tensors that runs on the GPU
Tensor Tensor::cuda_eq(const Tensor& other) {
    Tensor result(dimensions, true);

    int threads = 1024;
    int blocks = (total_elements + threads - 1) / threads;
    cuda_eq_kernel<<<blocks, threads>>>(data.get(), other.data.get(), result.data.get(), total_elements);
    

    return result;
}
