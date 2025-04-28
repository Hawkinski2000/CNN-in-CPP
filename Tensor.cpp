#include <iostream>
#include "Tensor.h"
using namespace std;


// Constructor for Tensor class
Tensor::Tensor(initializer_list<int> dims) : dimensions(dims) {}


// Function to return a tensor's shape vector
const vector<int>& Tensor::shape() const {
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
    Tensor x({2, 4, 8});

    vector<int> shape = x.shape();

    cout << "The first dim in the tensor x is " << shape[0] << endl;

    cout << "The tensor x has shape "<< x.shape_str() << endl;

    return 0;
}
