#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include "nn.h"
#include "Tensor.h"
#include "Node.h"
using namespace std;


// Constructor for the Linear class
Linear::Linear(size_t in_features, size_t out_features, bool use_bias) {
    this->in_features = in_features;
    this->out_features = out_features;
    weight = Tensor::rand({in_features, out_features});
    weight.requires_grad = true;
    this->use_bias = use_bias;
    if (use_bias) {
        bias = Tensor::rand({out_features}, in_features);
        bias.requires_grad = true;
    }
}

Tensor Linear::operator()(Tensor& input) {
    Tensor result = input.matmul(weight);
    if (use_bias) {
        result = result + bias;
    }
    return result;
}