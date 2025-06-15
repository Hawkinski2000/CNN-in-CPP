#include <iostream>
#include "Tensor.h"


__global__ void div_kernel(float* A, float* B, float* C, size_t total_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        C[idx] = A[idx] / B[idx];
    }
}

// Function for element-wise multiplication between tensors that runs on the GPU
Tensor Tensor::cuda_div(Tensor& other) {
    Tensor result(this->dimensions, true);
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    div_kernel<<<blocks, threads>>>(this->data.get(), other.data.get(), result.data.get(), result.total_elements);
    cudaDeviceSynchronize();
    return result;
}

// ---------------------------------------------------------------------------

__global__ void div_scalar_kernel(float* A, float* B, float value, size_t total_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        B[idx] = A[idx] / value;
    }
}

// Function for element-wise multiplication between tensors and scalars that runs on the GPU
Tensor Tensor::cuda_div_scalar(float value) {
    Tensor result(this->dimensions, true);
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    div_scalar_kernel<<<blocks, threads>>>(this->data.get(), result.data.get(), value, this->total_elements);
    cudaDeviceSynchronize();
    return result;
}

// ---------------------------------------------------------------------------

__global__ void div_inplace_kernel(float* A, float* B, size_t total_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        A[idx] /= B[idx];
    }
}

// Function for element-wise multiplication and assignment between tensors that runs on the GPU
Tensor& Tensor::cuda_div_(const Tensor& other) {
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    div_inplace_kernel<<<blocks, threads>>>(this->data.get(), other.data.get(), this->total_elements);
    cudaDeviceSynchronize();
    return *this;
}

// ---------------------------------------------------------------------------

__global__ void div_scalar_inplace_kernel(float* A, float value, size_t total_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        A[idx] /= value;
    }
}

// Function for element-wise multiplication and assignment between tensors and scalars that runs on the GPU
Tensor& Tensor::cuda_div_scalar_(float value) {
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    div_scalar_inplace_kernel<<<blocks, threads>>>(this->data.get(), value, this->total_elements);
    cudaDeviceSynchronize();
    return *this;
}
