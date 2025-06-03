#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include "nn.h"
#include "Tensor.h"
#include "Node.h"
using namespace std;


// Constructor for the Module class
Module::Module() = default;

// Destructor for the Module class
Module::~Module() = default;

// Overload the () operator to call forward() 
Tensor Module::operator()(Tensor input) {
    return forward(input);
}

// Function to return a vector containing the Module's parameters
vector<Tensor*> Module::parameters() {
    vector<Tensor*> parameters;
    for (Module* module : modules) {
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
