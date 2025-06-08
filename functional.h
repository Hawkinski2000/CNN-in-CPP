#pragma once
#include "Tensor.h"
using namespace std;


// Function to apply the rectified linear unit function to the input tensor
Tensor relu(const Tensor& input);

// Function to apply softmax to the input tensor
Tensor softmax(Tensor& input, optional<size_t> dim = nullopt);

// Function to apply log softmax to the input tensor
Tensor log_softmax(Tensor& input, optional<size_t> dim = nullopt);

// Function to compute the negative log likelihood loss from the input tensor and targets
Tensor nll_loss(Tensor& input, Tensor& targets);

// Function to compute the cross entropy loss between input tensor and targets
Tensor cross_entropy(Tensor& input, Tensor& targets);

// Function to extract sliding local blocks from a batched input tensor
Tensor unfold(Tensor& input, initializer_list<size_t> kernel_size, size_t dilation=1, size_t padding=0, size_t stride=1);

// Function to extract sliding local blocks from a batched input tensor that runs on the GPU.
Tensor unfold_cuda(Tensor input, size_t kH, size_t kW, size_t dilation=1, size_t padding=0, size_t stride=1);

// Function to combine an array of sliding local blocks into a large containing tensor
Tensor fold(Tensor& input, initializer_list<size_t> output_size, initializer_list<size_t> kernel_size, size_t dilation=1, size_t padding=0, size_t stride=1);
