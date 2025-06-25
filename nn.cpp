#include <iostream>
#include "nn.h"
#include "functional.h"
#include "Node.h"
using namespace std;


// Constructor for the Module class
Module::Module() = default;

// Destructor for the Module class
Module::~Module() = default;

vector<Module*> Module::modules;

// Overload the () operator to call forward() 
Tensor Module::operator()(Tensor input) {
    return forward(input);
}

// Function to return a vector containing the Module's parameters
vector<Tensor*> Module::parameters() {
    vector<Tensor*> parameters;
    for (Module* module : Module::modules) {
        vector<Tensor*> params = module->parameters();
        parameters.insert(parameters.end(), params.begin(), params.end());
    }
    return parameters;
}

// Function to return the type of Module
string Module::name() {
    return "Module";
}

// ---------------------------------------------------------------------------

// Constructor for the Linear class
Linear::Linear(size_t in_features, size_t out_features, bool use_bias) {
    this->in_features = in_features;
    this->out_features = out_features;
    weight = Tensor::rand({in_features, out_features}, in_features, true);
    weight.requires_grad = true;
    this->use_bias = use_bias;
    if (use_bias) {
        bias = Tensor::zeros({out_features}, true);
        bias.requires_grad = true;
    }
    Module::modules.push_back(this);
}

// Forwards inputs through the Linear layer
Tensor Linear::forward(Tensor& input) {
    Tensor result = input.matmul(weight);

    if (use_bias) {
        result = result + bias;
    }
    
    return result;
}

// Function to return a vector containing the Linear layer's parameters
vector<Tensor*> Linear::parameters() {
    if (use_bias) {
        return {&weight, &bias};
    }
    return {&weight};
}

// Function to return the type of Module
string Linear::name() {
    return "Linear";
}

// ---------------------------------------------------------------------------

// Constructor for the Conv2d class
Conv2d::Conv2d(size_t in_channels, size_t out_channels, initializer_list<size_t> kernel_size, size_t stride, size_t padding, size_t dilation, bool use_bias) {
    this->in_channels = in_channels;
    this->out_channels = out_channels;
    kH = *kernel_size.begin();
    if (kernel_size.size() == 1) {
        kW = kH;
    }
    else {
        kW = *(kernel_size.begin() + 1);
    }
    this->stride = stride;
    this->padding = padding;
    this->dilation = dilation; 
    weight = Tensor::rand({out_channels, in_channels, kH, kW}, in_channels * kH * kW, true);
    weight.requires_grad = true;
    this->use_bias = use_bias;
    if (use_bias) {
        bias = Tensor::zeros({1, out_channels, 1, 1}, true);
        bias.requires_grad = true;
    }
    Module::modules.push_back(this);
}

// Forwards inputs through the Conv2d layer
Tensor Conv2d::forward(Tensor& input) {
    Tensor inp_unf = unfold_cuda(input, {kH, kW}, dilation, padding, stride);
    Tensor w = weight.view({static_cast<int>(weight.shape()[0]), -1});
    Tensor out_unf = inp_unf.matmul(w, true, true, false);
    int N = input.shape()[0]; // Batch size
    int C = out_channels;
    size_t in_H = input.shape()[2]; // input height
    size_t in_W = input.shape()[3]; // input width
    int out_H = ((in_H + 2 * padding - dilation * (kH - 1) - 1) / stride) + 1; // output height
    int out_W = ((in_W + 2 * padding - dilation * (kW - 1) - 1) / stride) + 1; // output width
    Tensor result = out_unf.view({N, C, out_H, out_W});

    if (input.requires_grad) {
        result.node = make_shared<Conv2dBackward>(make_shared<Tensor>(input),
                                                  make_shared<Tensor>(weight),
                                                  make_shared<Tensor>(inp_unf),
                                                  initializer_list<size_t>{kH, kW},
                                                  stride,
                                                  padding,
                                                  dilation);
        result.node->tensor = &result;
    }

    if (use_bias) {
        result = result + bias;
    }

    return result;
}

// Function to return a vector containing the Conv2d layer's parameters
vector<Tensor*> Conv2d::parameters() {
    if (use_bias) {
        return {&weight, &bias};
    }
    return {&weight};
}

// Function to return the type of Module
string Conv2d::name() {
    return "Conv2d";
}

// ---------------------------------------------------------------------------

// Constructor for the MaxPool2d class
MaxPool2d::MaxPool2d(initializer_list<size_t> kernel_size, size_t stride, size_t padding, size_t dilation) {
    kH = *kernel_size.begin();
    if (kernel_size.size() == 1) {
        kW = kH;
    }
    else {
        kW = *(kernel_size.begin() + 1);
    }
    this->stride = stride;
    this->padding = padding;
    this->dilation = dilation;   
}

// Forwards inputs through the Conv2d layer
Tensor MaxPool2d::forward(Tensor& input) {
    Tensor result = maxpool2d_cuda(input, {kH, kW}, stride, padding, dilation);

    return result;
}

// Function to return the type of Module
string MaxPool2d::name() {
    return "MaxPool2d";
}
