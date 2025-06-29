#pragma once
#include "Tensor.h"
using namespace std;


class Tensor;

// Base class for all neural network modules
class Module {
    public:
        // Constructor for the Module class
        Module();

        // Destructor for the Module class
        virtual ~Module();

        // Forwards inputs through the Module
        virtual Tensor forward(Tensor& input) = 0;

        // Overload the () operator to call forward()
        Tensor operator()(Tensor input);

        // Function to return a vector containing the Module's parameters
        virtual vector<Tensor*> parameters();

        // Function to return the type of Module
        virtual string name();

        static vector<Module*> modules;
};

// ---------------------------------------------------------------------------

// A single layer of a neural network that inherits from the Module class
class Linear : public Module {
    public:
        // Constructor for the Linear class
        Linear(size_t in_features, size_t out_features, bool use_bias=true);

        // Forwards inputs through the Linear layer
        Tensor forward(Tensor& input) override;

        // Function to return a vector containing the Linear layer's parameters
        vector<Tensor*> parameters() override;

        // Function to return the type of Module
        string name() override;

        size_t in_features;
        size_t out_features;
        Tensor weight;
        bool use_bias;
        Tensor bias;
};

// ---------------------------------------------------------------------------

// A single 2D convolutional layer that inherits from the Module class
class Conv2d : public Module {
    public:
        // Constructor for the Conv2d class
        Conv2d(size_t in_channels, size_t out_channels, initializer_list<size_t> kernel_size, size_t stride=1, size_t padding=0, size_t dilation=1, bool use_bias=true);

        // Forwards inputs through the Conv2d layer
        Tensor forward(Tensor& input) override;

        // Function to return a vector containing the Conv2d layer's parameters
        vector<Tensor*> parameters() override;

        // Function to return the type of Module
        string name() override;

        size_t in_channels;
        size_t out_channels;
        size_t kH;
        size_t kW;
        size_t stride;
        size_t padding;
        size_t dilation;
        Tensor weight;
        bool use_bias;
        Tensor bias;
};

// ---------------------------------------------------------------------------

// A single 2D max pooling layer that inherits from the Module class
class MaxPool2d : public Module {
    public:
        // Constructor for the MaxPool2d class
        MaxPool2d(initializer_list<size_t> kernel_size, size_t stride=0, size_t padding=0, size_t dilation=1);

        // Forwards inputs through the MaxPool2d layer
        Tensor forward(Tensor& input) override;

        // Function to return the type of Module
        string name() override;

        size_t kH;
        size_t kW;
        size_t stride;
        size_t padding;
        size_t dilation;
        Tensor max_indices;
};

// ---------------------------------------------------------------------------

// A single rectified linear unit layer that inherits from the Module class
class ReLU : public Module {
    public:
        // Constructor for the ReLU class
        ReLU();

        // Forwards inputs through the ReLU layer
        Tensor forward(Tensor& input) override;

        // Function to return the type of Module
        string name() override;
};
