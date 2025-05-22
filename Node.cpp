#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include "Node.h"
#include "Tensor.h"
using namespace std;


// Constructor for the Node class
Node::Node() = default;

// Destructor for the Node class
Node::~Node() = default;

// Function to propagate gradients backward to child nodes
void Node::backward() {};

// ---------------------------------------------------------------------------

// Constructor for the AddBackward class
AddBackward::AddBackward(Tensor* a, Tensor* b) : lhs(a), rhs(b) {
    if (lhs->node) {
        children.push_back({lhs->node});
    }
    if (rhs->node) {
        children.push_back({rhs->node});
    }
}

// Function to propagate gradients of an addition node backward to child nodes
void AddBackward::backward() {
    for (size_t i = 0; i < tensor->numel(); i++) {
        lhs->grad.get()[i] += tensor->grad.get()[i];
        rhs->grad.get()[i] += tensor->grad.get()[i];
    }
}

// ---------------------------------------------------------------------------

// Constructor for the SubBackward class
SubBackward::SubBackward(Tensor* a, Tensor* b) : lhs(a), rhs(b) {
    if (lhs->node) {
        children.push_back({lhs->node});
    }
    if (rhs->node) {
        children.push_back({rhs->node});
    }
}

// Function to propagate gradients of a subtraction node backward to child nodes
void SubBackward::backward() {
    for (size_t i = 0; i < tensor->numel(); i++) {
        lhs->grad.get()[i] += tensor->grad.get()[i];
        rhs->grad.get()[i] -= tensor->grad.get()[i];
    }
}

// ---------------------------------------------------------------------------

// Constructor for the MulBackward class
MulBackward::MulBackward(Tensor* a, Tensor* b) : lhs(a), rhs(b) {
    if (lhs->node) {
        children.push_back({lhs->node});
    }
    if (rhs->node) {
        children.push_back({rhs->node});
    }
}

// Function to propagate gradients of a multiplication node backward to child nodes
void MulBackward::backward() {
    for (size_t i = 0; i < tensor->numel(); i++) {
        lhs->grad.get()[i] += tensor->grad.get()[i] * rhs->get_data()[i];
        rhs->grad.get()[i] += tensor->grad.get()[i] * lhs->get_data()[i];
    }
}

// ---------------------------------------------------------------------------

// Constructor for the DivBackward class
DivBackward::DivBackward(Tensor* a, Tensor* b) : lhs(a), rhs(b) {
    if (lhs->node) {
        children.push_back({lhs->node});
    }
    if (rhs->node) {
        children.push_back({rhs->node});
    }
}

// Function to propagate gradients of a division node backward to child nodes
void DivBackward::backward() {
    for (size_t i = 0; i < tensor->numel(); i++) {
        lhs->grad.get()[i] += tensor->grad.get()[i] * (1 / rhs->get_data()[i]);
        rhs->grad.get()[i] += tensor->grad.get()[i] * (-lhs->get_data()[i] / pow(rhs->get_data()[i], 2));
    }
    for (size_t i = 0; i < tensor->numel(); i++) {
        cout << lhs->grad.get()[i] << ", ";
    }
    cout << "and ";
    for (size_t i = 0; i < tensor->numel(); i++) {
        cout << rhs->grad.get()[i] << ", ";
    }
    cout << endl;
}