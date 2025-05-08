#include <iostream>
#include <memory>
#include "Tensor.h"
using namespace std;


/*
==============================================================================
TODO:
    - pow().
    - Arithmetic between tensors of different shapes using broadcasting rules.
    - squeeze()/unsqueeze().
    - min() and max().
    - sum().
    - mean().
    - exp().
    - log().
    - cat().
    - == operator.
    - Combine tensor-tensor and tensor-scalar arithmetic operators.
    - Overload the << operator for printing TensorSlice objects, e.g., a row.
    - Modify tensor() to take nested lists of values for creating tensors with
      multiple dimensions, e.g., tensor({{1, 2}, {3, 4}}).
    - matmul().

==============================================================================
*/

// Default constructor for the Tensor class
Tensor::Tensor() : total_elements(0) {};

// Copy constructor for the Tensor class
Tensor::Tensor(const Tensor& other) {
    dimensions = other.dimensions;
    total_elements = other.total_elements;
    strides = other.strides;
    data = make_shared<vector<float>>(*other.data);
}

// Move constructor for the Tensor class
Tensor::Tensor(Tensor&& other)
    : data(move(other.data)),
      dimensions(move(other.dimensions)),
      strides(move(other.strides)),
      total_elements(other.total_elements) {
    other.total_elements = 0;
}

// Constructor for Tensor class used by creation methods that specify a shape
Tensor::Tensor(initializer_list<size_t> dims) {
    dimensions.resize(dims.size());
    size_t i = 0;
    for (size_t dim : dims) {
        dimensions[i] = dim;
        i++;
    }

    // Calculate the total number of elements
    total_elements = 1;
    for (size_t dim : dims) {
        total_elements *= dim;
    }

    data = make_shared<vector<float>>(total_elements);

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

    data = make_shared<vector<float>>(values);
    dimensions = {values.size()};
    strides = {1};
    total_elements = values.size();
}

// Constructor for Tensor class used by tensor() to convert a vector to a tensor
Tensor::Tensor(const vector<float>& values) {
    
    // Handle empty vector case
    if (values.empty()) {
        dimensions = {0};
        total_elements = 0;
        strides = {1};
        return;
    }

    data = make_shared<vector<float>>(values);
    dimensions = {values.size()};
    strides = {1};
    total_elements = values.size();
}

// Function to create an empty tensor from a specified shape
Tensor Tensor::empty(initializer_list<size_t> dims) {
    return Tensor(dims);
}

// Function to create a tensor of zeros from a specified shape
Tensor Tensor::zeros(initializer_list<size_t> dims) {
    Tensor tensor(dims);
    fill(tensor.data->begin(), tensor.data->end(), 0);
    return tensor;
}

// Function to create a tensor of ones from a specified shape
Tensor Tensor::ones(initializer_list<size_t> dims) {
    Tensor tensor(dims);
    fill(tensor.data->begin(), tensor.data->end(), 1);
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
Tensor::TensorSlice Tensor::operator[](size_t index) {
    if (index >= total_elements) {
        throw out_of_range("Index out of bounds");
    }
    return TensorSlice(data, dimensions, strides, index * strides[0], 1);
}

// Overload the = operator to move a temporary tensor into an existing tensor
Tensor& Tensor::operator=(Tensor&& other) {
    if (this != &other) {
        data = move(other.data);
        dimensions = move(other.dimensions);
        strides = move(other.strides);
        total_elements = other.total_elements;
        other.total_elements = 0;
    }
    return *this;
}

// Overload the = operator for copying one tensor to another
Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        dimensions = other.dimensions;
        strides = other.strides;
        total_elements = other.total_elements;
        data = make_shared<vector<float>>(*other.data);
    }
    return *this;
}

// Function to return a new tensor with the same data as the original tensor but of a different shape
Tensor Tensor::view(initializer_list<int> shape) {
    Tensor result;

    // The new tensor shares the same data as the original tensor
    result.data = data;

    // Calculate dimensions of the new tensor
    vector<size_t> dims(shape.size());
    size_t product = 1;
    int infer_idx = -1;
    int i = 0;
    for (int dim : shape) {
        if (dim == -1) {
            if (infer_idx > -1) {
                throw runtime_error("only one dimension can be inferred");
            }
            infer_idx = i;
        }
        else {
            product *= dim;
            dims[i] = dim;
        }
        i++;
    }
    if (infer_idx > -1) {
        if (total_elements % product > 0) {
            throw runtime_error("Shape is invalid for input size " + to_string(total_elements));
        }
        dims[infer_idx] = total_elements / product;
    }
    else if (product != total_elements) {
        throw runtime_error("Shape is invalid for input size " + to_string(total_elements));
    }
    result.dimensions = move(dims);

    // Calculate strides of the new tensor from its dimensions
    result.strides.resize(result.dimensions.size());
    size_t stride = 1;
    for (size_t i = result.dimensions.size(); i-- > 0;) {
        result.strides[i] = stride;
        stride *= result.dimensions[i];
    }

    result.total_elements = total_elements;

    return result;
}

// Function to flatten a tensor by reshaping it into a one-dimensional tensor
Tensor Tensor::flatten() {
    return this->view({-1});
}

// Function to return a tensor that is a transposed version of a tensor
Tensor Tensor::transpose(size_t dim0, size_t dim1) {
    Tensor result;

    // The new tensor shares the same data as the original tensor
    result.data = data;

    // Get lengths of specified dimensions
    size_t size0 = dimensions[dim0];
    size_t size1 = dimensions[dim1];

    result.dimensions = dimensions;

    // Swap the specified dimensions in the new tensor
    result.dimensions[dim0] = size1;
    result.dimensions[dim1] = size0;

    // Calculate strides of the new tensor from its dimensions
    result.strides.resize(result.dimensions.size());
    size_t stride = 1;
    for (size_t i = result.dimensions.size(); i-- > 0;) {
        result.strides[i] = stride;
        stride *= result.dimensions[i];
    }

    result.total_elements = total_elements;

    return result;
}

// Function to return the sum of all elements in a tensor
Tensor Tensor::sum() {
    Tensor result = tensor({0});
    for (float value : *data) {
        result[0] += value;
    }
    return result;
}

// Overload the + operator for element-wise addition between tensors
Tensor Tensor::operator+(const Tensor& other) {
    Tensor result = *this;
    for (size_t i = 0; i < other.total_elements; i++) {
        (*result.data)[i] += (*other.data)[i];
    }
    return result;
}

// Overload the + operator for element-wise addition between tensors and scalars
Tensor Tensor::operator+(float value) {
    Tensor result = *this;
    for (size_t i = 0; i < total_elements; i++) {
        (*result.data)[i] += value;
    }
    return result;
}

// Overload the += operator for element-wise addition and assignment between tensors
Tensor& Tensor::operator+=(const Tensor& other) {
    for (size_t i = 0; i < other.total_elements; i++) {
        (*data)[i] += (*other.data)[i];
    }
    return *this;
}

// Overload the += operator for element-wise addition and assignment between tensors and scalars
Tensor& Tensor::operator+=(float value) {
    for (size_t i = 0; i < total_elements; i++) {
        (*data)[i] += value;
    }
    return *this;
}

// Overload the - operator for element-wise subtraction between tensors
Tensor Tensor::operator-(const Tensor& other) {
    Tensor result = *this;
    for (size_t i = 0; i < other.total_elements; i++) {
        (*result.data)[i] -= (*other.data)[i];
    }
    return result;
}

// Overload the - operator for element-wise subtraction between tensors and scalars
Tensor Tensor::operator-(float value) {
    Tensor result = *this;
    for (size_t i = 0; i < total_elements; i++) {
        (*result.data)[i] -= value;
    }
    return result;
}

// Overload the -= operator for element-wise subtraction and assignment between tensors
Tensor& Tensor::operator-=(const Tensor& other) {
    for (size_t i = 0; i < other.total_elements; i++) {
        (*data)[i] -= (*other.data)[i];
    }
    return *this;
}

// Overload the -= operator for element-wise subtraction and assignment between tensors and scalars
Tensor& Tensor::operator-=(float value) {
    for (size_t i = 0; i < total_elements; i++) {
        (*data)[i] -= value;
    }
    return *this;
}

// Overload the * operator for element-wise multiplication between tensors
Tensor Tensor::operator*(const Tensor& other) {
    Tensor result = *this;
    for (size_t i = 0; i < other.total_elements; i++) {
        (*result.data)[i] *= (*other.data)[i];
    }
    return result;
}

// Overload the * operator for element-wise multiplication between tensors and scalars
Tensor Tensor::operator*(float value) {
    Tensor result = *this;
    for (size_t i = 0; i < total_elements; i++) {
        (*result.data)[i] *= value;
    }
    return result;
}

// Overload the *= operator for element-wise multiplication and assignment between tensors
Tensor& Tensor::operator*=(const Tensor& other) {
    for (size_t i = 0; i < other.total_elements; i++) {
        (*data)[i] *= (*other.data)[i];
    }
    return *this;
}

// Overload the *= operator for element-wise multiplication and assignment between tensors and scalars
Tensor& Tensor::operator*=(float value) {
    for (size_t i = 0; i < total_elements; i++) {
        (*data)[i] *= value;
    }
    return *this;
}

// Overload the / operator for element-wise division between tensors
Tensor Tensor::operator/(const Tensor& other) {
    Tensor result = *this;
    for (size_t i = 0; i < other.total_elements; i++) {
        (*result.data)[i] /= (*other.data)[i];
    }
    return result;
}

// Overload the / operator for element-wise division between tensors and scalars
Tensor Tensor::operator/(float value) {
    Tensor result = *this;
    for (size_t i = 0; i < total_elements; i++) {
        (*result.data)[i] /= value;
    }
    return result;
}

// Overload the /= operator for element-wise division and assignment between tensors
Tensor& Tensor::operator/=(const Tensor& other) {
    for (size_t i = 0; i < other.total_elements; i++) {
        (*data)[i] /= (*other.data)[i];
    }
    return *this;
}

// Overload the /= operator for element-wise division and assignment between tensors and scalars
Tensor& Tensor::operator/=(float value) {
    for (size_t i = 0; i < total_elements; i++) {
        (*data)[i] /= value;
    }
    return *this;
}

// Overload the << operator for printing the contents of a tensor
ostream& operator<<(ostream& os, const Tensor& tensor) {
    os << '(';
    for (size_t i = 0; i < tensor.total_elements; i++) {
        os << (*tensor.data)[i];
        if (i < tensor.total_elements - 1) {
            os << ", ";
        }
    }
    os << ')';
    return os;
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

// ---------------------------------------------------------------------------

// Constructor for TensorSlice class used for chaining multiple [] operators
Tensor::TensorSlice::TensorSlice(shared_ptr<vector<float>> data, const vector<size_t>& dimensions,
    const vector<size_t>& strides, size_t offset, size_t level)
        : data(data),
          dimensions(dimensions),
          strides(strides),
          offset(offset),
          level(level) {}

// Overload the [] operator for indexing into the tensor data
Tensor::TensorSlice Tensor::TensorSlice::operator[](size_t index) {
    if (level >= dimensions.size()) {
        throw out_of_range("Index out of bounds");
    }
    size_t new_offset = offset + index * strides[level];
    return TensorSlice(data, dimensions, strides, new_offset, level + 1);
}

// Overload the = operator for assigning values to indices in the tensor
Tensor::TensorSlice& Tensor::TensorSlice::operator=(float value) {
    (*data)[offset] = value;
    return *this;
}

// Overload the float reference conversion operator to return data after the final []
Tensor::TensorSlice::operator float&() {
    if (level != dimensions.size()) {
        throw out_of_range("Cannot convert TensorSlice to float reference");
    }
    return (*data)[offset];
}

// ---------------------------------------------------------------------------


int main() {
    Tensor a = Tensor::tensor({2, 4});
    cout << "The first element in the tensor a is " << a[0] << endl;
    cout << "The tensor a has shape " << a.shape_str() << endl << endl;

    vector<float> v = {2, 4};
    Tensor b = Tensor::tensor(v);
    cout << "The first element in the tensor b is " << b[0] << endl;
    cout << "The tensor b has shape " << b.shape_str() << endl << endl;

    Tensor c = Tensor::empty({2, 2});
    cout << "The first element in the tensor c is " << c[0][0] << endl;
    cout << "The tensor c has shape " << c.shape_str() << endl << endl;

    Tensor d = Tensor::zeros({2, 16});
    cout << "The first element in the tensor d is " << d[0][0] << endl;
    cout << "The tensor d has shape " << d.shape_str() << endl << endl;

    Tensor e = Tensor::ones({8, 4});
    cout << "The first element in the tensor e is " << e[0][0] << endl;
    cout << "The tensor e has shape " << e.shape_str() << endl << endl;

    cout << "The tensor e contains: " << endl;
    cout << e << endl << endl;

    Tensor f = e + e;
    cout << "The sum of tensor e with itself is tensor f, which contains: " << endl;
    cout << f << endl << endl;

    Tensor g = e - f;
    cout << "The difference between tensor e and tensor f is tensor g, which contains: " << endl;
    cout << g << endl << endl;

    Tensor h = f * f;
    cout << "The product of tensor f with itself is tensor h, which contains: " << endl;
    cout << h << endl << endl;

    Tensor i = e / h;
    cout << "The quotient of tensor e and tensor h is tensor i, which contains: " << endl;
    cout << i << endl << endl;

    i[0][0] = 2;
    cout << "After assigning the value 2 to the first index in tensor i, it now contains: " << endl;
    cout << i << endl << endl;

    Tensor j; 
    j = Tensor::ones({4, 4});
    cout << "The tensor j created using the default constructor and move assignment operator contains: " << endl;
    cout << j << endl << endl;

    cout << "The product of tensor h with itself is a tensor containing: " << endl;
    cout << h * h << endl << endl;

    Tensor k = j + 2.5;
    cout << "The sum of tensor j and 2.5 is tensor k, which contains: " << endl;
    cout << k << endl << endl;

    Tensor l = j - 2.5;
    cout << "The difference between tensor j and 2.5 is tensor l, which contains: " << endl;
    cout << l << endl << endl;

    Tensor m = k * 2;
    cout << "The product of tensor k and 2 is tensor m, which contains: " << endl;
    cout << m << endl << endl;

    Tensor n = l / 2;
    cout << "The quotient of tensor l and 2 is tensor n, which contains: " << endl;
    cout << n << endl << endl;

    m += k;
    cout << "After adding tensor k to tensor m, tensor m now contains: " << endl;
    cout << m << endl << endl;

    m -= j;
    cout << "After subtracting tensor j from tensor m, tensor m now contains: " << endl;
    cout << m << endl << endl;

    m *= n;
    cout << "After multiplying tensor m by tensor n, tensor m now contains: " << endl;
    cout << m << endl << endl;

    m /= l;
    cout << "After dividing tensor m by tensor l, tensor m now contains: " << endl;
    cout << m << endl << endl;

    m += 1;
    cout << "After adding 1 to tensor m, it now contains: " << endl;
    cout << m << endl << endl;

    m -= 2.5;
    cout << "After subtracting 2.5 from tensor m, it now contains: " << endl;
    cout << m << endl << endl;

    m *= 2;
    cout << "After multiplying tensor m by 2, it now contains: " << endl;
    cout << m << endl << endl;

    m /= 3;
    cout << "After dividing tensor m by 3, it now contains: " << endl;
    cout << m << endl << endl;

    cout << "The tensor m has shape " << m.shape_str() << endl << endl;
    Tensor o = m.view({2, 2, 4});
    cout << "The tensor m was reshaped with view() to a new tensor o with shape " << o.shape_str() << endl << endl;

    cout << "The tensor o has shape " << o.shape_str() << endl << endl;
    Tensor p = o.view({2, 2, 2, 2});
    cout << "The tensor o was reshaped with view() to a new tensor p with shape " << p.shape_str() << endl << endl;

    cout << "The tensor p has shape " << p.shape_str() << endl << endl;
    Tensor q = p.view({8, -1});
    cout << "The tensor p was reshaped with view() to a new tensor q with shape " << q.shape_str() << endl << endl;

    cout << "The tensor q has shape " << q.shape_str() << endl << endl;
    q = q.flatten();
    cout << "The tensor q was flattened and now has the shape " << q.shape_str() << endl << endl;

    cout << "The tensor o has shape " << o.shape_str() << endl << endl;
    o = o.transpose(0, 2);
    cout << "The tensor o was transposed along dimensions 0 and 2 and now has the shape " << o.shape_str() << endl << endl;

    cout << "The tensor j contains " << j << endl << endl;
    j = j.sum();
    cout << "After applying sum to tensor j, it now contains " << j << endl << endl;

    return 0;
}
