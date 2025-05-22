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

// Function to propagate gradients backward to child nodes
void AddBackward::backward() {
    lhs->grad += tensor->grad;
    rhs->grad += tensor->grad;
    cout << lhs->grad << " and " << rhs->grad << endl;
}
