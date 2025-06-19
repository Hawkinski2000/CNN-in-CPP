#include <iostream>
#include "Tensor.h"


// Kernel for nll_loss_cuda()
__global__ void nll_loss_cuda_kernel(const float* __restrict__ input,
                                     const float* __restrict__ targets,
                                     float* __restrict__ output,
                                     size_t batch_size,
                                     size_t num_classes) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size) {
        size_t target_class = static_cast<size_t>(targets[idx]);
        size_t input_idx = idx * num_classes + target_class;
        float loss = -input[input_idx];
        atomicAdd(output, loss);
    }
}

// Kernel to normalize loss for nll_loss_cuda()
__global__ void normalize_loss_kernel(float* loss, float batch_size) {
    *loss /= batch_size;
}

// Function to compute the negative log likelihood loss from the input tensor and targets that runs on the GPU.
Tensor nll_loss_cuda(Tensor& input, Tensor& targets) {
    Tensor result = Tensor::zeros({1}, true);

    size_t batch_size;
    size_t num_classes;
    if (input.dimensions.size() < 2) {
        batch_size = 1;
        num_classes = input.dimensions[0];
    }
    else {
        batch_size = input.dimensions[0];
        num_classes = input.dimensions[1];
    }

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    nll_loss_cuda_kernel<<<blocks, threads>>>(input.data.get(),
                                              targets.data.get(),
                                              result.data.get(),
                                              batch_size,
                                              num_classes);

    cudaDeviceSynchronize();

    normalize_loss_kernel<<<1, 1>>>(result.data.get(), static_cast<float>(batch_size));
    
    cudaDeviceSynchronize();

    return result;
}
