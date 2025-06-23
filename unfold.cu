#include <iostream>
#include "Tensor.h"


__global__ void unfold_kernel(const float* input,
                              float* output,
                              int N,
                              int C,
                              int in_H,
                              int in_W,
                              int kH,
                              int kW,
                              int out_H,
                              int out_W,
                              int stride,
                              int padding,
                              int dilation,
                              int out_L) {
    int n = blockIdx.x; // Batch index
    int l = blockIdx.y; // Output column
    int patch_idx = threadIdx.x; // Index within each patch vector

    if (patch_idx >= C * kH * kW) {
        return;
    }

    int out_h = l / out_W; // output height position
    int out_w = l % out_W; // output width position

    int c = patch_idx / (kH * kW); // Input channel index
    int kh = (patch_idx / kW) % kH; // Kernel height position
    int kw = patch_idx % kW; // Kernel width position

    int in_h = out_h * stride - padding + kh * dilation; // input height position
    int in_w = out_w * stride - padding + kw * dilation; // input width position

    float val = 0;
    if (in_h >= 0 && in_h < in_H && in_w >= 0 && in_w < in_W) {
        int input_idx = ((n * C + c) * in_H + in_h) * in_W + in_w; // Index in the input tensor
        val = input[input_idx];
    }

    int out_idx = ((n * (C * kH * kW) + patch_idx) * out_L) + l; // Index in the unfolded output tensor
    output[out_idx] = val;
}

Tensor unfold_cuda(Tensor input, initializer_list<size_t> kernel_size, size_t dilation, size_t padding, size_t stride) {
    size_t kH = *kernel_size.begin(); // Kernel height
    size_t kW; // Kernel width
    if (kernel_size.size() == 1) {
        kW = kH;
    }
    else {
        kW = *(kernel_size.begin() + 1);
    }

    size_t N = input.dimensions[0]; // Batch size

    size_t C = input.dimensions[1]; // in_channels
    
    size_t in_H = input.dimensions[2]; // input height
    size_t in_W = input.dimensions[3]; // input width

    size_t out_H = ((in_H + 2 * padding - dilation * (kH - 1) - 1) / stride) + 1; // output height
    size_t out_W = ((in_W + 2 * padding - dilation * (kW - 1) - 1) / stride) + 1; // output width
    size_t L = out_H * out_W; // Total number of blocks

    // Create a result tensor with its data on the GPU
    Tensor result = Tensor::empty({N, C * kH * kW, L}, true);

    dim3 gridDim(static_cast<unsigned int>(N), static_cast<unsigned int>(L));
    dim3 blockDim(static_cast<unsigned int>(C * kH * kW));

    unfold_kernel<<<gridDim, blockDim>>>(input.data.get(),
                                         result.data.get(),
                                         N,
                                         C,
                                         in_H,
                                         in_W,
                                         kH,
                                         kW,
                                         out_H,
                                         out_W,
                                         stride,
                                         padding,
                                         dilation,
                                         L);

    

    return result;
}
