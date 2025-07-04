#include <iostream>
#include "Tensor.h"


__global__ void fold_kernel(const float* input,
                            float* output,
                            int N,
                            int C,
                            int out_H,
                            int out_W,
                            int kH,
                            int kW,
                            int stride,
                            int padding,
                            int dilation,
                            int L,
                            int conv_out_H,
                            int conv_out_W) {
    int n = blockIdx.x; // Batch index
    int l = blockIdx.y; // Output column
    int patch_idx = threadIdx.x; // Index within each patch vector

    if (patch_idx >= C * kH * kW) {
        return;
    }

    // The top-left corner of this patch
    int out_i = l / conv_out_H;
    int out_j = l % conv_out_W;

    int c = patch_idx / (kH * kW); // Input channel index
    int kh = (patch_idx / kW) % kH; // Kernel height position
    int kw = patch_idx % kW; // Kernel width position

    int out_h = out_i * stride - padding + kh * dilation; // output height position
    int out_w = out_j * stride - padding + kw * dilation; // output width position

    if (out_h < 0 || out_h >= out_H || out_w < 0 || out_w >= out_W) {
        return;
    }

    int input_idx = ((n * (C * kH * kW) + patch_idx) * L) + l; // Index in the input tensor
    int output_idx = ((n * C + c) * out_H + out_h) * out_W + out_w; // Index in the folded output tensor

    atomicAdd(&output[output_idx], input[input_idx]);
}

Tensor fold_cuda(Tensor& input, initializer_list<size_t> output_size, initializer_list<size_t> kernel_size, size_t dilation, size_t padding, size_t stride) {
    size_t kH = *kernel_size.begin(); // Kernel height
    size_t kW; // Kernel width
    if (kernel_size.size() == 1) {
        kW = kH;
    }
    else {
        kW = *(kernel_size.begin() + 1);
    }

    size_t out_H = *output_size.begin(); // Output height
    size_t out_W = *(output_size.begin() + 1); // Output width

    size_t conv_out_H = (out_H + 2 * padding - dilation * (kH - 1) - 1) / stride + 1;
    size_t conv_out_W = (out_W + 2 * padding - dilation * (kW - 1) - 1) / stride + 1;

    size_t N = input.dimensions[0]; // Batch size

    size_t C = input.dimensions[1] / (kH * kW); // Channels

    size_t L = input.dimensions[2]; // Total number of blocks

    Tensor result = Tensor::zeros({N, C, out_H, out_W}, true);

    dim3 gridDim(static_cast<unsigned int>(N), static_cast<unsigned int>(L));
    dim3 blockDim(static_cast<unsigned int>(C * kH * kW));

    fold_kernel<<<gridDim, blockDim>>>(input.data.get(),
                                       result.data.get(),
                                       N,
                                       C,
                                       out_H,
                                       out_W,
                                       kH,
                                       kW,
                                       stride,
                                       padding,
                                       dilation,
                                       L,
                                       conv_out_H,
                                       conv_out_W);

    

    return result;
}
