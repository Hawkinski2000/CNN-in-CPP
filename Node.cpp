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
AddBackward::AddBackward(shared_ptr<Tensor> a, shared_ptr<Tensor> b) : lhs(make_shared<Tensor>(*a)), rhs(b) {
    if (lhs->node) {
        children.push_back({lhs->node});
    }
    if (rhs->node) {
        children.push_back({rhs->node});
    }
}

// Function to propagate gradients of an addition node backward to child nodes
void AddBackward::backward() {
    for (size_t i = 0; i < tensor->total_elements; i++) {
        lhs->grad.get()[i] += tensor->grad.get()[i];
        rhs->grad.get()[i] += tensor->grad.get()[i];
    }

    // cout << "===============================================================================" << endl;
    // cout << "An AddBackward node has the gradients:" << endl;
    // for (size_t i = 0; i < lhs->total_elements; i++) {
    //     cout << lhs->grad.get()[i] << ", ";
    // }
    // cout << "and ";
    // for (size_t i = 0; i < rhs->total_elements; i++) {
    //     cout << rhs->grad.get()[i] << ", ";
    // }
    // cout << endl << "===============================================================================" << endl;
}

// ---------------------------------------------------------------------------

// Constructor for the SubBackward class
SubBackward::SubBackward(shared_ptr<Tensor> a, shared_ptr<Tensor> b) : lhs(make_shared<Tensor>(*a)), rhs(b) {
    if (lhs->node) {
        children.push_back({lhs->node});
    }
    if (rhs->node) {
        children.push_back({rhs->node});
    }
}

// Function to propagate gradients of a subtraction node backward to child nodes
void SubBackward::backward() {
    for (size_t i = 0; i < tensor->total_elements; i++) {
        lhs->grad.get()[i] += tensor->grad.get()[i];
        rhs->grad.get()[i] -= tensor->grad.get()[i];
    }
}

// ---------------------------------------------------------------------------

// Constructor for the MulBackward class
MulBackward::MulBackward(shared_ptr<Tensor> a, shared_ptr<Tensor> b) : lhs(make_shared<Tensor>(*a)), rhs(b) {
    if (lhs->node) {
        children.push_back({lhs->node});
    }
    if (rhs->node) {
        children.push_back({rhs->node});
    }
}

// Function to propagate gradients of a multiplication node backward to child nodes
void MulBackward::backward() {
    for (size_t i = 0; i < tensor->total_elements; i++) {
        lhs->grad.get()[i] += tensor->grad.get()[i] * rhs->get_data()[i];
        rhs->grad.get()[i] += tensor->grad.get()[i] * lhs->get_data()[i];
    }
}

// ---------------------------------------------------------------------------

// Constructor for the DivBackward class
DivBackward::DivBackward(shared_ptr<Tensor> a, shared_ptr<Tensor> b) : lhs(make_shared<Tensor>(*a)), rhs(b) {
    if (lhs->node) {
        children.push_back({lhs->node});
    }
    if (rhs->node) {
        children.push_back({rhs->node});
    }
}

// Function to propagate gradients of a division node backward to child nodes
void DivBackward::backward() {
    for (size_t i = 0; i < tensor->total_elements; i++) {
        lhs->grad.get()[i] += tensor->grad.get()[i] * (1 / rhs->get_data()[i]);
        rhs->grad.get()[i] += tensor->grad.get()[i] * (-lhs->get_data()[i] / pow(rhs->get_data()[i], 2));
    }
}

// ---------------------------------------------------------------------------

// Constructor for the MatmulBackward class
MatmulBackward::MatmulBackward(shared_ptr<Tensor> a, shared_ptr<Tensor> b) : lhs(make_shared<Tensor>(*a)), rhs(b) {
    if (lhs->node) {
        children.push_back({lhs->node});
    }
    if (rhs->node) {
        children.push_back({rhs->node});
    }
}

// Function to propagate gradients of a matmul node backward to child nodes
void MatmulBackward::backward() {
    Tensor dLdc = *tensor;
    copy(tensor->grad.get(), tensor->grad.get() + tensor->total_elements, dLdc.data.get());

    Tensor dLda = dLdc.matmul(*rhs, false, true, false);
    Tensor dLdb = lhs->matmul(dLdc, true, false, false);

    size_t A_batch_count = 1;
    if (lhs->dimensions.size() > 2) {
        for (size_t i = 0; i < lhs->dimensions.size() - 2; i++) {
            A_batch_count *= lhs->dimensions[i];
        }
    } else {
        A_batch_count = 1;
    }
    size_t B_batch_count = 1;
    if (rhs->dimensions.size() > 2) {
        for (size_t i = 0; i < rhs->dimensions.size() - 2; i++) {
            B_batch_count *= rhs->dimensions[i];
        }
    } else {
        B_batch_count = 1;
    }
    size_t C_batch_count = 1;
    if (tensor->dimensions.size() > 2) {
        for (size_t i = 0; i < tensor->dimensions.size() - 2; i++) {
            C_batch_count *= tensor->dimensions[i];
        }
    } else {
        C_batch_count = 1;
    }

    size_t m = lhs->dimensions[lhs->dimensions.size() - 2];
    size_t k = lhs->dimensions[lhs->dimensions.size() - 1];
    size_t n = rhs->dimensions[rhs->dimensions.size() - 1];
    // A and B were both broadcast
    if (C_batch_count > A_batch_count && C_batch_count > B_batch_count) {
        if (lhs->requires_grad) {
            for (size_t i = 0; i < C_batch_count * m * k; i++) {
                lhs->grad.get()[i % lhs->total_elements] += dLda.data.get()[i % lhs->total_elements];
            }
        }
        if (rhs->requires_grad) {
            for (size_t i = 0; i < C_batch_count * k * n; i++) {
                rhs->grad.get()[i % rhs->total_elements] += dLdb.data.get()[i % rhs->total_elements];
            }
        }
    }
    // Only B was broadcast
    else if (A_batch_count > B_batch_count) {
        if (lhs->requires_grad) {
            for (size_t i = 0; i < lhs->total_elements; i++) {
                lhs->grad.get()[i] += dLda.data.get()[i];
            }
        }
        if (rhs->requires_grad) {
            for (size_t i = 0; i < C_batch_count * k * n; i++) {
                rhs->grad.get()[i % rhs->total_elements] += dLdb.data.get()[i % rhs->total_elements];
            }
        }
    }
    // Only A was broadcast
    else if (B_batch_count > A_batch_count) {
        if (lhs->requires_grad) {
            for (size_t i = 0; i < C_batch_count * m * k; i++) {
                lhs->grad.get()[i % lhs->total_elements] += dLda.data.get()[i % lhs->total_elements];
            }
        }
        if (rhs->requires_grad) {
            for (size_t i = 0; i < rhs->total_elements; i++) {
                rhs->grad.get()[i] += dLdb.data.get()[i];
            }
        }
    }
    else {
        if (lhs->requires_grad) {
            for (size_t i = 0; i < lhs->total_elements; i++) {
                lhs->grad.get()[i] += dLda.data.get()[i];
            }
        }
        if (rhs->requires_grad) {
            for (size_t i = 0; i < rhs->total_elements; i++) {
                rhs->grad.get()[i] += dLdb.data.get()[i];
            }
        }
    }

    // cout << "===============================================================================" << endl;
    // cout << "A MatmulBackward node has the gradients:" << endl;
    // for (size_t i = 0; i < lhs->total_elements; i++) {
    //     cout << lhs->grad.get()[i] << ", ";
    // }
    // cout << "and ";
    // for (size_t i = 0; i < rhs->total_elements; i++) {
    //     cout << rhs->grad.get()[i] << ", ";
    // }
    // cout << endl << "===============================================================================" << endl;
}
