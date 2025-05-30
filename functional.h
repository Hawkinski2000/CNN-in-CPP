#pragma once
#include <optional>
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
