#include <iostream>
#include "Tensor.h"
using namespace std;

// TODO: Remove constructor, use only static functions for creating
// tensors, e.g. tensor(), empty(), zeros(), ones(), rand(), etc.

// Constructor for Tensor class
Tensor::Tensor(initializer_list<size_t> dims) : dimensions(dims) {
    // Calculate the total number of elements
    total_elements = 1;
    for (size_t dim : dims) {
        total_elements *= dim;
    }

    // Allocate memory for tensor
    data = new float[total_elements];

    // Calculate strides from tensor dimensions
    strides.resize(dimensions.size());
    size_t stride = 1;
    for (size_t i = dimensions.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= dimensions[i];
    }
}

// Destructor for Tensor class
Tensor::~Tensor() {
    delete[] data;
}

// Function to return a tensor's shape vector
const vector<size_t>& Tensor::shape() const {
    return dimensions;
}

// Function to return a tensor's shape as a string
string Tensor::shape_str() {
    string shape = "(";
    for (size_t i = 0; i < dimensions.size(); i++) {
        shape += to_string(dimensions[i]);
        if (i < dimensions.size() - 1) {
            shape += ", ";
        } 
    }
    shape += ')';
    return shape;
}

int main() {
    Tensor x{2, 4, 8, 16};

    cout << "The first dim in the tensor x is " << x.shape()[0] << endl;

    cout << "The tensor x has shape "<< x.shape_str() << endl;

    return 0;
}
