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

        virtual string name() {
            return "Node";
        }

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

        // Function to propagate gradients of an addition node backward to child nodes
        void backward() override;

        string name() override {
            return "AddBackward";
        }
};

// ---------------------------------------------------------------------------

// Node for subtraction in the automatic differentiation graph that inherits from the Node class
class SubBackward : public Node {
    shared_ptr<Tensor> lhs;
    shared_ptr<Tensor> rhs;

    public:
        // Constructor for the SubBackward class
        SubBackward(shared_ptr<Tensor> a, shared_ptr<Tensor> b);

        // Function to propagate gradients of a subtraction node backward to child nodes
        void backward() override;

        string name() override {
            return "SubBackward";
        }
};

// ---------------------------------------------------------------------------

// Node for multiplication in the automatic differentiation graph that inherits from the Node class
class MulBackward : public Node {
    shared_ptr<Tensor> lhs;
    shared_ptr<Tensor> rhs;

    public:
        // Constructor for the MulBackward class
        MulBackward(shared_ptr<Tensor> a, shared_ptr<Tensor> b);

        // Function to propagate gradients of a multiplication node backward to child nodes
        void backward() override;

        string name() override {
            return "MulBackward";
        }
};

// ---------------------------------------------------------------------------

// Node for division in the automatic differentiation graph that inherits from the Node class
class DivBackward : public Node {
    shared_ptr<Tensor> lhs;
    shared_ptr<Tensor> rhs;

    public:
        // Constructor for the DivBackward class
        DivBackward(shared_ptr<Tensor> a, shared_ptr<Tensor> b);

        // Function to propagate gradients of a division node backward to child nodes
        void backward() override;

        string name() override {
            return "DivBackward";
        }
};

// ---------------------------------------------------------------------------

// Node for matmul in the automatic differentiation graph that inherits from the Node class
class MatmulBackward : public Node {
    shared_ptr<Tensor> lhs;
    shared_ptr<Tensor> rhs;

    public:
        // Constructor for the MatmulBackward class
        MatmulBackward(shared_ptr<Tensor> a, shared_ptr<Tensor> b);

        // Function to propagate gradients of a matmul node backward to child nodes
        void backward() override;

        string name() override {
            return "MatmulBackward";
        }
};
