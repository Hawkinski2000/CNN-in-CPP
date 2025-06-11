#pragma once
#include <vector>
#include <memory>
using namespace std;


class Tensor;

// Base class for nodes in the automatic differentiation graph
class Node {
    public:
        // Destructor for the Node class
        virtual ~Node();

        // Function to propagate gradients backward to child nodes
        virtual void backward() = 0;

        // Function to return the type of Node
        virtual string name() = 0;

        Tensor* tensor = nullptr;
        vector<shared_ptr<Node>> children;
};

// ---------------------------------------------------------------------------

// Node for addition in the automatic differentiation graph that inherits from the Node class
class AddBackward : public Node {
    shared_ptr<Tensor> lhs;
    shared_ptr<Tensor> rhs;

    public:
        // Constructor for the AddBackward class
        AddBackward(shared_ptr<Tensor> a, shared_ptr<Tensor> b);

        // Function to propagate gradients backward to child nodes
        void backward() override;

        // Function to return the type of Node
        string name() override;
};

// ---------------------------------------------------------------------------

// Node for subtraction in the automatic differentiation graph that inherits from the Node class
class SubBackward : public Node {
    shared_ptr<Tensor> lhs;
    shared_ptr<Tensor> rhs;

    public:
        // Constructor for the SubBackward class
        SubBackward(shared_ptr<Tensor> a, shared_ptr<Tensor> b);

        // Function to propagate gradients backward to child nodes
        void backward() override;

        // Function to return the type of Node
        string name() override;
};

// ---------------------------------------------------------------------------

// Node for multiplication in the automatic differentiation graph that inherits from the Node class
class MulBackward : public Node {
    shared_ptr<Tensor> lhs;
    shared_ptr<Tensor> rhs;

    public:
        // Constructor for the MulBackward class
        MulBackward(shared_ptr<Tensor> a, shared_ptr<Tensor> b);

        // Function to propagate gradients backward to child nodes
        void backward() override;

        // Function to return the type of Node
        string name() override;
};

// ---------------------------------------------------------------------------

// Node for division in the automatic differentiation graph that inherits from the Node class
class DivBackward : public Node {
    shared_ptr<Tensor> lhs;
    shared_ptr<Tensor> rhs;

    public:
        // Constructor for the DivBackward class
        DivBackward(shared_ptr<Tensor> a, shared_ptr<Tensor> b);

        // Function to propagate gradients backward to child nodes
        void backward() override;

        // Function to return the type of Node
        string name() override;
};

// ---------------------------------------------------------------------------

// Node for matmul in the automatic differentiation graph that inherits from the Node class
class MatmulBackward : public Node {
    shared_ptr<Tensor> lhs;
    shared_ptr<Tensor> rhs;

    public:
        // Constructor for the MatmulBackward class
        MatmulBackward(shared_ptr<Tensor> a, shared_ptr<Tensor> b);

        // Function to propagate gradients backward to child nodes
        void backward() override;

        // Function to return the type of Node
        string name() override;
};

// ---------------------------------------------------------------------------

// Node for ReLU in the automatic differentiation graph that inherits from the Node class
class ReLUBackward : public Node {
    shared_ptr<Tensor> input;

    public:
        // Constructor for the ReLUBackward class
        ReLUBackward(shared_ptr<Tensor> input);

        // Function to propagate gradients backward to child nodes
        void backward() override;

        // Function to return the type of Node
        string name() override;
};

// ---------------------------------------------------------------------------


// Node for log softmax in the automatic differentiation graph that inherits from the Node class
class LogSoftmaxBackward : public Node {
    shared_ptr<Tensor> input;
    shared_ptr<Tensor> softmax_values;

    public:
        // Constructor for the LogSoftmaxBackward class
        LogSoftmaxBackward(shared_ptr<Tensor> input, shared_ptr<Tensor> softmax_values);

        // Function to propagate gradients backward to child nodes
        void backward() override;

        // Function to return the type of Node
        string name() override;
};

// ---------------------------------------------------------------------------

// Node for negative log likelihood loss in the automatic differentiation graph that inherits from the Node class
class NLLLossBackward : public Node {
    shared_ptr<Tensor> input;
    shared_ptr<Tensor> targets;

    public:
        // Constructor for the NLLLossBackward class
        NLLLossBackward(shared_ptr<Tensor> input, shared_ptr<Tensor> targets);

        // Function to propagate gradients backward to child nodes
        void backward() override;

        // Function to return the type of Node
        string name() override;
};

// ---------------------------------------------------------------------------

// Node for a 2D convolutional layer in the automatic differentiation graph that inherits from the Node class
class Conv2dBackward : public Node {
    shared_ptr<Tensor> input;
    shared_ptr<Tensor> weight;
    shared_ptr<Tensor> inp_unf;
    vector<size_t> kernel_size;
    size_t stride;
    size_t padding;
    size_t dilation;

    public:
        // Constructor for the Conv2dBackward class
        Conv2dBackward(shared_ptr<Tensor> input,
                       shared_ptr<Tensor> weight,
                       shared_ptr<Tensor> inp_unf,
                       initializer_list<size_t> kernel_size,
                       size_t stride,
                       size_t padding,
                       size_t dilation);

        // Function to propagate gradients backward to child nodes
        void backward() override;

        // Function to return the type of Node
        string name() override;
};

// ---------------------------------------------------------------------------

// Node for a 2D max pooling layer in the automatic differentiation graph that inherits from the Node class
class MaxPool2dBackward : public Node {
    shared_ptr<Tensor> input;
    shared_ptr<Tensor> max_indices;
    vector<size_t> kernel_size;
    size_t stride;
    size_t padding;
    size_t dilation;

    public:
        // Constructor for the MaxPool2dBackward class
        MaxPool2dBackward(shared_ptr<Tensor> input,
                          shared_ptr<Tensor> max_indices,
                          initializer_list<size_t> kernel_size,
                          size_t stride,
                          size_t padding,
                          size_t dilation);

        // Function to propagate gradients backward to child nodes
        void backward() override;

        // Function to return the type of Node
        string name() override;
};
