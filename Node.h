#pragma once
#include <vector>
#include <memory>
#include "Tensor.h"
using namespace std;


class Tensor;

// Base class for nodes in the automatic differentiation graph
class Node {
    public:
        // Constructor for the Node class
        Node();

        // Destructor for the Node class
        virtual ~Node();

        // Function to propagate gradients backward to child nodes
        virtual void backward();

        Tensor* tensor;
        vector<shared_ptr<Node>> children;
};

// ---------------------------------------------------------------------------

// Node for addition in the automatic differentiation graph that inherits from the Node class
class AddBackward : public Node {
    Tensor* lhs;
    Tensor* rhs;

    public:
        // Constructor for the AddBackward class
        AddBackward(Tensor* a, Tensor* b);

        // Function to propagate gradients of an addition node backward to child nodes
        void backward() override;
};

// ---------------------------------------------------------------------------

// Node for subtraction in the automatic differentiation graph that inherits from the Node class
class SubBackward : public Node {
    Tensor* lhs;
    Tensor* rhs;

    public:
        // Constructor for the SubBackward class
        SubBackward(Tensor* a, Tensor* b);

        // Function to propagate gradients of a subtraction node backward to child nodes
        void backward() override;
};
