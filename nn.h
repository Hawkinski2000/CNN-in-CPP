#pragma once
#include <vector>
#include <memory>
#include "Tensor.h"
using namespace std;


class Tensor;

// A single layer of a neural network
class Linear {
    public:
        // Constructor for the Linear class
        Linear(size_t in_features, size_t out_features, bool use_bias=true);

        Tensor operator()(Tensor& input);

        size_t in_features;
        size_t out_features;
        Tensor weight;
        bool use_bias;
        Tensor bias;
};