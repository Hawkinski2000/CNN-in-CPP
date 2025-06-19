#include <iostream>
#include "Tensor.h"


// Kernel for relu_cuda()
__global__ void relu_kernel(const float* __restrict__ input, float* output, size_t total_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// Function to apply the rectified linear unit function to the input tensor that runs on the GPU
Tensor relu_cuda(const Tensor& input) {
    Tensor result(input.dimensions, true);
    int threads = 256;
    int blocks = (input.total_elements + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(input.data.get(),
                                     result.data.get(),
                                     input.total_elements);

    cudaDeviceSynchronize();

    return result;
}
