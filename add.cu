#include <iostream>
#include "Tensor.h"


// Kernel for cuda_add()
__global__ void add_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, size_t total_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        C[idx] = A[idx] + B[idx];
    }
}

// Function for element-wise addition between tensors that runs on the GPU
Tensor Tensor::cuda_add(Tensor& other) {
    Tensor result(this->dimensions, true);
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(this->data.get(), other.data.get(), result.data.get(), result.total_elements);
    cudaDeviceSynchronize();
    return result;
}

// ---------------------------------------------------------------------------

// Kernel for cuda_add_scalar()
__global__ void add_scalar_kernel(float* __restrict__ A, float* __restrict__ B, float value, size_t total_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        B[idx] = A[idx] + value;
    }
}

// Function for element-wise addition between tensors and scalars that runs on the GPU
Tensor Tensor::cuda_add_scalar(float value) {
    Tensor result(this->dimensions, true);
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    add_scalar_kernel<<<blocks, threads>>>(this->data.get(), result.data.get(), value, this->total_elements);
    cudaDeviceSynchronize();
    return result;
}

// ---------------------------------------------------------------------------

// Kernel for cuda_add_()
__global__ void add_inplace_kernel(float* A, float* B, size_t total_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        A[idx] += B[idx];
    }
}

// Function for element-wise addition and assignment between tensors that runs on the GPU
Tensor& Tensor::cuda_add_(const Tensor& other) {
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    add_inplace_kernel<<<blocks, threads>>>(this->data.get(), other.data.get(), this->total_elements);
    cudaDeviceSynchronize();
    return *this;
}

// ---------------------------------------------------------------------------

// Kernel for cuda_add_scalar_()
__global__ void add_scalar_inplace_kernel(float* A, float value, size_t total_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        A[idx] += value;
    }
}

// Function for element-wise addition and assignment between tensors and scalars that runs on the GPU
Tensor& Tensor::cuda_add_scalar_(float value) {
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    add_scalar_inplace_kernel<<<blocks, threads>>>(this->data.get(), value, this->total_elements);
    cudaDeviceSynchronize();
    return *this;
}
