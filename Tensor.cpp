#include <iostream>
#include "Tensor.h"
using namespace std;

// // TODO: Copy and move constructors for copying and assigning tensors,
// // e.g. "Tensor a = b;" and "c = b;"

// Constructor for Tensor class used by creation methods that specify a shape
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

// Constructor for Tensor class used by tensor() where values are specified
Tensor::Tensor(initializer_list<float> values) {

    // Handle empty values case
    if (values.size() == 0) {
        dimensions = {0};
        total_elements = 0;
        strides = {1};
        return;
    }

    total_elements = values.size();
    data = new float[total_elements];
    dimensions = {1};
    strides = {1};

    // Copy the values into the tensor's data
    copy(values.begin(), values.end(), data);
}

// Constructor for Tensor class used by tensor() to convert a vector to a tensor
Tensor::Tensor(vector<float>& values) {
    
    // Handle empty vector case
    if (values.empty()) {
        dimensions = {0};
        total_elements = 0;
        strides = {1};
        return;
    }

    total_elements = values.size();
    data = new float[total_elements];
    dimensions = {1};
    strides = {1};

    // Copy the values into the tensor's data
    copy(values.begin(), values.end(), data);
}

// Destructor for Tensor class
Tensor::~Tensor() {
    delete[] data;
}

// Function to create an empty tensor from a specified shape
Tensor Tensor::empty(initializer_list<size_t> dims) {
    return Tensor(dims);
}

// Function to create a tensor of zeros from a specified shape
Tensor Tensor::zeros(initializer_list<size_t> dims) {
    Tensor tensor(dims);
    fill(tensor.data, tensor.data + tensor.total_elements, 0);
    return tensor;
}

// Function to create a tensor of ones from a specified shape
Tensor Tensor::ones(initializer_list<size_t> dims) {
    Tensor tensor(dims);
    fill(tensor.data, tensor.data + tensor.total_elements, 1);
    return tensor;
}

// Function to create a tensor from specified values
Tensor Tensor::tensor(initializer_list<float> values) {
    return Tensor(values);
}

// Function to create a tensor from a vector
Tensor Tensor::tensor(vector<float>& values) {
    return Tensor(values);
}

// Overload the [] operator for indexing into the tensor data
float& Tensor::operator[](size_t index) {
    if (index >= total_elements) {
        throw out_of_range("Index out of bounds");
    }
    return data[index];
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
    Tensor a = Tensor::tensor({2, 4, 8, 16});
    cout << "The first element in the tensor a is " << a[0] << endl;
    cout << "The tensor a has shape "<< a.shape_str() << endl << endl;

    vector<float> v = {2, 4, 8, 16};
    Tensor b = Tensor::tensor(v);
    cout << "The first element in the tensor b is " << b[0] << endl;
    cout << "The tensor b has shape "<< b.shape_str() << endl << endl;

    Tensor c = Tensor::empty({3, 2});
    cout << "The first element in the tensor c is " << c[0] << endl;
    cout << "The tensor c has shape "<< c.shape_str() << endl << endl;

    Tensor d = Tensor::zeros({8, 4});
    cout << "The first element in the tensor d is " << d[0] << endl;
    cout << "The tensor d has shape "<< d.shape_str() << endl << endl;

    Tensor e = Tensor::ones({2, 4, 6});
    cout << "The first element in the tensor e is " << e[0] << endl;
    cout << "The tensor e has shape "<< e.shape_str() << endl << endl;

    return 0;
}
