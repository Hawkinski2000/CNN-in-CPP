#include <iostream>
#include "nn.h"
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
    weight = Tensor::rand({in_features, out_features});
    weight.requires_grad = true;
    this->use_bias = use_bias;
    if (use_bias) {
        bias = Tensor::rand({out_features}, in_features);
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
Conv2d::Conv2d(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride, size_t padding, bool use_bias) {
    this->in_channels = in_channels;
    this->out_channels = out_channels;
    this->kernel_size = kernel_size;
    this->stride = stride;
    this->padding = padding;
    weight = Tensor::rand({out_channels, in_channels, kernel_size, kernel_size});
    weight.requires_grad = true;
    this->use_bias = use_bias;
    if (use_bias) {
        bias = Tensor::rand({out_channels}, in_channels);
        bias.requires_grad = true;
    }
    Module::modules.push_back(this);
}

// Forwards inputs through the Conv2d layer
Tensor Conv2d::forward(Tensor& input) {
    Tensor result = input.matmul(weight);
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
