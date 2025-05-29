#pragma once
#include <vector>
#include <memory>
#include "Tensor.h"
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
