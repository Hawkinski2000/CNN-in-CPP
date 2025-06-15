#include <iostream>
#include "cuda_utils.cuh"
#include "Tensor.h"


__global__ void fill_kernel(float* data, size_t total_elements, float value) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        data[idx] = value;
    }
}

void cuda_fill(float* data, size_t total_elements, float value) {
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    fill_kernel<<<blocks, threads>>>(data, total_elements, value);
    cudaDeviceSynchronize();
}
