#include <iostream>
#include "Tensor.h"
#include "Node.h"


__global__ void maxpool2d_kernel(const float* input,
                                 float* output,
                                 float* max_indices,
                                 int N,
                                 int C,
                                 int in_H,
                                 int in_W,
                                 int out_H,
                                 int out_W,
                                 int kH,
                                 int kW,
                                 int stride,
                                 int padding,
                                 int dilation) {
    int n = blockIdx.x; // Batch index
    int c = blockIdx.y; // Input channel index

    int out_h = threadIdx.y; // Output height position
    int out_w = threadIdx.x; // Output width position

    if (out_h >= out_H || out_w >= out_W) {
        return;
    }

    // The top-left corner of this patch
    int in_i = out_h * stride - padding;
    int in_j = out_w * stride - padding;

    float max_val = -INFINITY;
    int max_idx;

    for (int kh = 0; kh < kH; kh++) {
        for (int kw = 0; kw < kW; kw++) {
            int in_h = in_i + kh * dilation; // Input height position
            int in_w = in_j + kw * dilation; // Input width position

            if (in_h >= 0 && in_h < in_H && in_w >= 0 && in_w < in_W) {
                int input_idx = ((n * C + c) * in_H + in_h) * in_W + in_w;
                if (input[input_idx] > max_val) {
                    max_val = input[input_idx];
                    max_idx = input_idx;
                }
            }
        }
    }

    int output_idx = ((n * C + c) * out_H + out_h) * out_W + out_w;
    output[output_idx] = max_val;
    max_indices[output_idx] = max_idx;
}

Tensor maxpool2d_cuda(Tensor input, initializer_list<size_t> kernel_size, size_t stride, size_t padding, size_t dilation) {
    size_t kH = *kernel_size.begin(); // Kernel height
    size_t kW; // Kernel width
    if (kernel_size.size() == 1) {
        kW = kH;
    }
    else {
        kW = *(kernel_size.begin() + 1);
    }

    if (stride == 0) {
        stride = kH;
    }
    
    size_t N = input.dimensions[0]; // Batch size

    size_t C = input.dimensions[1]; // Channels

    size_t in_H = input.dimensions[2]; // Input height
    size_t in_W = input.dimensions[3]; // Input width

    size_t out_H = ((in_H + 2 * padding - dilation * (kH - 1) - 1) / stride) + 1; // Output height
    size_t out_W = ((in_W + 2 * padding - dilation * (kW - 1) - 1) / stride) + 1; // Output width

    Tensor result = Tensor::empty({N, C, out_H, out_W}, true);
    Tensor max_indices = Tensor::empty({N, C, out_H, out_W}, true);

    dim3 gridDim(N, C);
    dim3 blockDim(out_W, out_H);

    // Allocate GPU memory for the input tensor if not already
    if (!input.device_data) {
        cudaMalloc(&input.device_data, input.total_elements * sizeof(float));
    }

    // Transfer the input tensor's data from CPU to GPU
    cudaMemcpy(input.device_data, input.data.get(), sizeof(float) * input.total_elements, cudaMemcpyHostToDevice);

    maxpool2d_kernel<<<gridDim, blockDim>>>(input.device_data,
                                            result.device_data,
                                            max_indices.device_data,
                                            N,
                                            C,
                                            in_H,
                                            in_W,
                                            out_H,
                                            out_W,
                                            kH,
                                            kW,
                                            stride,
                                            padding,
                                            dilation);

    cudaDeviceSynchronize();

    // Transfer the result tensor's data from GPU to CPU
    cudaMemcpy(result.data.get(), result.device_data, sizeof(float) * result.total_elements, cudaMemcpyDeviceToHost);

    // Transfer the max_indices tensor's data from GPU to CPU
    cudaMemcpy(max_indices.data.get(), max_indices.device_data, sizeof(float) * max_indices.total_elements, cudaMemcpyDeviceToHost);

    if (input.requires_grad) {
        result.node = make_shared<MaxPool2dBackward>(make_shared<Tensor>(input),
                                                     make_shared<Tensor>(max_indices),
                                                     initializer_list<size_t>{kH, kW},
                                                     stride,
                                                     padding,
                                                     dilation);
        result.node->tensor = &result;
    }

    return result;
}
