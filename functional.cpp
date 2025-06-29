#include <iostream>
#include "functional.h"
#include "Node.h"
#include <cuda_runtime.h>
using namespace std;

// Function to apply the rectified linear unit function to the input tensor
Tensor relu(const Tensor& input) {
    Tensor result;

    if (input.device == "cuda") {
        result = relu_cuda(input);
    }
    else {
        result = Tensor(input.dimensions);
        for (size_t i = 0; i < input.total_elements; i++) {
            result.data.get()[i] = max(0.0f, input.data.get()[i]);
        }
    }

    if (input.requires_grad) {
        result.node = make_shared<ReLUBackward>(make_shared<Tensor>(input));
        result.node->tensor = &result;
    }

    return result;
}

// Function to apply softmax to the input tensor
Tensor softmax(Tensor& input, optional<size_t> dim) {
    Tensor max = input.max(dim);
    Tensor shifted_values = input - max;
    Tensor exp_values = shifted_values.exp();
    Tensor sum_exp_values = exp_values.sum(dim);
    Tensor result = exp_values / sum_exp_values;
    return result;
}

// Function to apply log softmax to the input tensor
Tensor log_softmax(Tensor& input, optional<size_t> dim) {
    Tensor max = input.max(dim);
    input.requires_grad = false;
    Tensor shifted_values = input - max;
    input.requires_grad = true;
    Tensor exp_values = shifted_values.exp();
    Tensor sum_exp_values = exp_values.sum(dim);
    Tensor log_sum_exp_values = sum_exp_values.log();
    
    shifted_values.requires_grad = false;
    Tensor result = shifted_values - log_sum_exp_values;

    exp_values.requires_grad = false;
    Tensor softmax_values = exp_values / sum_exp_values;

    if (input.requires_grad) {
        result.node = make_shared<LogSoftmaxBackward>(make_shared<Tensor>(input), make_shared<Tensor>(softmax_values));
        result.node->tensor = &result;
    }

    return result;
}

// Function to compute the negative log likelihood loss from the input tensor and targets
Tensor nll_loss(Tensor& input, Tensor& targets) {
    Tensor result;
    
    if (input.device == "cuda") {
        result = nll_loss_cuda(input, targets);
    }
    else {
        result = Tensor::zeros({1});
        
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
        
        for (size_t i = 0; i < batch_size; i++) {
            size_t class_idx = static_cast<size_t>(targets[i]);
            size_t flat_index = i * num_classes + class_idx;
            result[0] -= input.data.get()[flat_index];
        }
        result[0] /= batch_size;
    }

    if (input.requires_grad) {
        result.node = make_shared<NLLLossBackward>(make_shared<Tensor>(input), make_shared<Tensor>(targets));
        result.node->tensor = &result;
    }

    return result;
}

// Function to compute the cross entropy loss between input tensor and targets
Tensor cross_entropy(Tensor& input, Tensor& targets) {
    Tensor log_probs = log_softmax(input, 1);
    Tensor loss = nll_loss(log_probs, targets);
    return loss;
}

// Function to extract sliding local blocks from a batched input tensor
Tensor unfold(Tensor& input, initializer_list<size_t> kernel_size, size_t dilation, size_t padding, size_t stride) {
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

    Tensor result = Tensor::empty({N, (C * kH * kW), L});

    for (size_t n = 0; n < N; n++) { // Sample in batch
        size_t col_idx = 0; // Column index in the unfolded output
        for (size_t out_h = 0; out_h < out_H; out_h++) { // Output height position
            for (size_t out_w = 0; out_w < out_W; out_w++) { // Output width position
                size_t patch_idx = 0; // Index within each patch vector
                for (size_t c = 0; c < C; c++) { // Input channel index
                    for (size_t kh = 0; kh < kH; kh++) { // Kernel height position
                        for (size_t kw = 0; kw < kW; kw++) { // Kernel width position
                            int in_h = static_cast<int>(out_h * stride - padding + kh * dilation); // Row in input
                            int in_w = static_cast<int>(out_w * stride - padding + kw * dilation); // Column in input

                            if (in_h >= 0 && in_h < in_H && in_w >= 0 && in_w < in_W) {
                                result[n][patch_idx][col_idx] = input[n][c][in_h][in_w];
                            }
                            else {
                                result[n][patch_idx][col_idx] = 0; // Zero padding
                            }
                            
                            patch_idx++;
                        }
                    }
                }
                col_idx++;
            }
        }
    }
    return result;
}

// Function to combine an array of sliding local blocks into a large containing tensor
Tensor fold(Tensor& input, initializer_list<size_t> output_size, initializer_list<size_t> kernel_size, size_t dilation, size_t padding, size_t stride) {
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

    size_t N = input.dimensions[0]; // Batch size

    size_t C = input.dimensions[1] / (kH * kW); // Channels

    size_t L = input.dimensions[2]; // Total number of blocks

    Tensor result = Tensor::zeros({N, C, out_H, out_W});

    for (size_t n = 0; n < N; n++) {
        for (size_t l = 0; l < L; l++) {
            // The top-left corner of this patch
            size_t out_i = l / ((out_W - kW + 2 * padding) / stride + 1);
            size_t out_j = l % ((out_W - kW + 2 * padding) / stride + 1);

            for (size_t c = 0; c < C; c++) { // Input channel index
                for (size_t kh = 0; kh < kH; kh++) { // Kernel height position
                    for (size_t kw = 0; kw < kW; kw++) { // Kernel width position
                        size_t out_h = out_i * stride - padding + kh * dilation; // output height position
                        size_t out_w = out_j * stride - padding + kw * dilation; // output width position
                        
                        if (out_h < out_H && out_w < out_W) {
                            size_t patch_idx = c * kH * kW + kh * kW + kw; // Index within each patch vector
                            float value = input[n][patch_idx][l];
                            result[n][c][out_h][out_w] += value;
                        }
                    }
                }
            }
        }
    }
    return result;
}
