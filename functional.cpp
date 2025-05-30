#include <iostream>
#include "functional.h"
#include "Tensor.h"
using namespace std;


// Function to apply the rectified linear unit function to the input tensor
Tensor relu(const Tensor& input) {
    Tensor result = Tensor(input.dimensions);
    for (size_t i = 0; i < input.total_elements; i++) {
        result.data.get()[i] = max(0.0f, input.data.get()[i]);
    }
    return result;
}

// Function to apply softmax to the input tensor
Tensor softmax(Tensor& input, optional<size_t> dim) {
    Tensor max = input.max();
    Tensor shifted_values = input - max;
    Tensor exp_values = shifted_values.exp();
    Tensor sum_exp_values = exp_values.sum(dim);
    Tensor result = exp_values / sum_exp_values;
    return result;
}

// Function to apply log softmax to the input tensor
Tensor log_softmax(Tensor& input, optional<size_t> dim) {
    Tensor max = input.max();
    Tensor shifted_values = input - max;
    Tensor exp_values = shifted_values.exp();
    Tensor sum_exp_values = exp_values.sum(dim);
    Tensor log_sum_exp_values = sum_exp_values.log();
    Tensor result = shifted_values - log_sum_exp_values;
    return result;
}

// Function to compute the negative log likelihood loss from the input tensor and targets
Tensor nll_loss(Tensor& input, Tensor& targets) {
    Tensor result = Tensor::zeros({1});
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
    return result;
}

// Function to compute the cross entropy loss between input tensor and targets
Tensor cross_entropy(Tensor& input, Tensor& targets) {
    Tensor log_probs = log_softmax(input, 1);
    Tensor loss = nll_loss(log_probs, targets);
    return loss;
}
