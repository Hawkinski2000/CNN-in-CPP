#include <iostream>
#include <vector>
#include <memory>
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
    for (size_t i = 0; i < tensor->numel(); i++) {
        cout << lhs->grad.get()[i] << ", ";
    }
    cout << "and ";
    for (size_t i = 0; i < tensor->numel(); i++) {
        cout << rhs->grad.get()[i] << ", ";
    }
    cout << endl;
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
