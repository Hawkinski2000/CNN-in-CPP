#include <iostream>
#include <memory>
#include "Tensor.h"
#include "Engine.h"
using namespace std;


// Compile with: nvcc -o Tensor Tensor.cpp Node.cpp Engine.cpp matmul.cu -lcublas

/*
==============================================================================
TODO:
    - Multiple Tensor data types.
    - pow() (floats and negatives as exponents).
    - squeeze()/unsqueeze().
    - min() and max() (specifying dimensions to reduce).
    - sum() (specifying dimensions to reduce).
    - mean() (specifying dimensions to reduce).
    - exp().
    - log().
    - cat().
    - == operator.
    - Combine tensor-tensor and tensor-scalar arithmetic operators.
    - Overload the << operator for printing TensorSlice objects, e.g., a row.
    - Modify tensor() to take nested lists of values for creating tensors with
      multiple dimensions, e.g., tensor({{1, 2}, {3, 4}}).

==============================================================================
*/

// Default constructor for the Tensor class
Tensor::Tensor() : total_elements(0) {};

// Copy constructor for the Tensor class
Tensor::Tensor(const Tensor& other) {
    dimensions = other.dimensions;
    total_elements = other.total_elements;
    strides = other.strides;    
    data = shared_ptr<float>(new float[other.total_elements], default_delete<float[]>());
    copy(other.data.get(), other.data.get() + other.total_elements, data.get());
    grad = shared_ptr<float>(new float[total_elements], default_delete<float[]>());
    fill(grad.get(), grad.get() + total_elements, 0);
}

// Move constructor for the Tensor class
Tensor::Tensor(Tensor&& other)
    : data(move(other.data)),
      dimensions(move(other.dimensions)),
      strides(move(other.strides)),
      total_elements(other.total_elements) {
    other.total_elements = 0;
    grad = shared_ptr<float>(new float[total_elements], default_delete<float[]>());
    fill(grad.get(), grad.get() + total_elements, 0);
}

// Constructor for Tensor class used by creation methods that specify a shape as an initializer_list
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

    data = shared_ptr<float>(new float[total_elements], default_delete<float[]>());

    // Calculate strides from tensor dimensions
    strides = compute_strides(dimensions);

    grad = shared_ptr<float>(new float[total_elements], default_delete<float[]>());
    fill(grad.get(), grad.get() + total_elements, 0);
}

// Constructor for Tensor class used by creation methods that specify a shape as a vector
Tensor::Tensor(const vector<size_t>& dims) {
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

    data = shared_ptr<float>(new float[total_elements], default_delete<float[]>());

    // Calculate strides from tensor dimensions
    strides = compute_strides(dimensions);

    grad = shared_ptr<float>(new float[total_elements], default_delete<float[]>());
    fill(grad.get(), grad.get() + total_elements, 0);
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

    data = shared_ptr<float>(new float[values.size()], default_delete<float[]>());
    copy(values.begin(), values.end(), data.get());

    dimensions = {values.size()};
    strides = {1};
    total_elements = values.size();

    grad = shared_ptr<float>(new float[total_elements], default_delete<float[]>());
    fill(grad.get(), grad.get() + total_elements, 0);
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

    data = shared_ptr<float>(new float[values.size()], default_delete<float[]>());
    copy(values.begin(), values.end(), data.get());

    dimensions = {values.size()};
    strides = {1};
    total_elements = values.size();

    grad = shared_ptr<float>(new float[total_elements], default_delete<float[]>());
    fill(grad.get(), grad.get() + total_elements, 0);
}

// ---------------------------------------------------------------------------

// Function to create an empty tensor from a shape specified as an initializer_list
Tensor Tensor::empty(initializer_list<size_t> dims) {
    return Tensor(dims);
}

// Function to create an empty tensor from a shape specified as a vector
Tensor Tensor::empty(vector<size_t> dims) {
    return Tensor(dims);
}

// Function to create a tensor of zeros from a specified shape
Tensor Tensor::zeros(initializer_list<size_t> dims) {
    Tensor tensor(dims);
    fill(tensor.data.get(), tensor.data.get() + tensor.total_elements, 0);
    return tensor;
}

// Function to create tensor of zeros from a shape specified as a vector
Tensor Tensor::zeros(vector<size_t> dims) {
    Tensor tensor(dims);
    fill(tensor.data.get(), tensor.data.get() + tensor.total_elements, 0);
    return tensor;
}

// Function to create a tensor of ones from a specified shape
Tensor Tensor::ones(initializer_list<size_t> dims) {
    Tensor tensor(dims);
    fill(tensor.data.get(), tensor.data.get() + tensor.total_elements, 1);
    return tensor;
}

// Function to create a tensor of ones from a shape specified as a vector
Tensor Tensor::ones(vector<size_t> dims) {
    Tensor tensor(dims);
    fill(tensor.data.get(), tensor.data.get() + tensor.total_elements, 1);
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

// ---------------------------------------------------------------------------

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
        grad = move(other.grad);
    }
    return *this;
}

// Overload the = operator for copying one tensor to another
Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        dimensions = other.dimensions;
        strides = other.strides;
        total_elements = other.total_elements;
        data = shared_ptr<float>(new float[other.total_elements], default_delete<float[]>());
        copy(other.data.get(), other.data.get() + other.total_elements, data.get());
        grad = shared_ptr<float>(new float[total_elements], default_delete<float[]>());
        copy(other.grad.get(), other.grad.get() + other.total_elements, grad.get());
    }
    return *this;
}

// ---------------------------------------------------------------------------

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
    result.strides = compute_strides(result.dimensions);

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
    result.strides = compute_strides(result.dimensions);

    result.total_elements = total_elements;

    return result;
}

// ---------------------------------------------------------------------------

// Function to return the sum of all elements in a tensor
Tensor Tensor::sum() {
    Tensor result = tensor({0});
    for (size_t i = 0; i < total_elements; i++) {
        result[0] += data.get()[i];
    }
    return result;
}

// Function to return the mean of all elements in a tensor
Tensor Tensor::mean() {
    Tensor result = tensor({0});
    for (size_t i = 0; i < total_elements; i++) {
        result[0] += data.get()[i];
    }
    result[0] /= total_elements;
    return result;
}

// Function for element-wise powers between tensors and scalars
Tensor Tensor::pow(int exponent) {
    Tensor result = *this;
    float value;
    for (size_t i = 0; i < total_elements; i++) {
        value = result.data.get()[i];
        for (int j = 0; j < exponent - 1; j++) {
            result.data.get()[i] *= value;    
        }
    }
    return result;
}

// Function to return the minimum value of all elements in a tensor
Tensor Tensor::min() {
    Tensor result = empty({1});
    float min = data.get()[0];
    for (size_t i = 1; i < total_elements; i++) {
        min = std::min(min, data.get()[i]);
    }
    result[0] = min;
    return result;
}

// Function to return the maximum value of all elements in a tensor
Tensor Tensor::max() {
    Tensor result = empty({1});
    float max = data.get()[0];
    for (size_t i = 1; i < total_elements; i++) {
        max = std::max(max, data.get()[i]);
    }
    result[0] = max;
    return result;
}

// Function to return true if two tensors have the same shape and elements, otherwise false.
bool Tensor::equal(const Tensor& other) {
    if (shape() == other.shape()) {
        for (size_t i = 0; i < total_elements; i++) {
            if (data.get()[i] != other.data.get()[i]) {
                return false;
            } 
        }
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------

// Function to compute a tensor's strides from its dimensions
vector<size_t> Tensor::compute_strides(const vector<size_t>& dimensions) {
    vector<size_t> strides(dimensions.size());
    size_t stride = 1;
    for (size_t i = dimensions.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= dimensions[i];
    }
    return strides;
}

// Function to pad a shape or strides vector with 1's or 0's for broadcasting
vector<size_t> Tensor::pad_vector(const vector<size_t>& original, size_t num_dims, size_t pad_with) {
    vector<size_t> padded(num_dims);

    // Pad original with the specified number to make compatible for broadcasting
    size_t pad_len = num_dims - original.size();
    fill(padded.begin(), padded.begin() + pad_len, pad_with);
    copy(original.begin(), original.end(), padded.begin() + pad_len);

    return padded;
}

// Function to return a vector containing the resulting tensor shape from an operation on two broadcastable tensors
vector<size_t> Tensor::broadcast_result_shape(const vector<size_t>& a_dims, const vector<size_t>& b_dims) {
    size_t num_dims = std::max(a_dims.size(), b_dims.size());

    vector<size_t> padded_a_dims = pad_vector(a_dims, num_dims, 1);
    vector<size_t> padded_b_dims = pad_vector(b_dims, num_dims, 1);

    vector<size_t> result_dims(num_dims);
    for (size_t i = 0; i < num_dims; i++) {
        if (padded_a_dims[i] == padded_b_dims[i] || padded_a_dims[i] == 1 || padded_b_dims[i] == 1) {
            result_dims[i] = std::max(padded_a_dims[i], padded_b_dims[i]);
        }
        else {
            throw runtime_error("Shapes are not broadcastable.");
        }
    }
    return result_dims;
}

// Function to return a vector containing the broadcastable strides of a tensor given a resulting tensor shape
vector<size_t> Tensor::broadcast_strides(const vector<size_t>& original_dims, const vector<size_t>& original_strides, const vector<size_t>& result_dims) {
    size_t num_dims = result_dims.size();

    vector<size_t> padded_dims = pad_vector(original_dims, num_dims, 1);
    vector<size_t> padded_strides = pad_vector(original_strides, num_dims, 0);
    
    vector<size_t> broadcasted_strides(num_dims);
    for (size_t i = 0; i < num_dims; i++) {
        if (padded_dims[i] == 1 && result_dims[i] != 1) {
            broadcasted_strides[i] = 0;
        }
        else {
            broadcasted_strides[i] = padded_strides[i];
        }
    }
    return broadcasted_strides;
}

// Function to compute the memory offset for a multi-dimensional index from the tensor's strides
size_t Tensor::compute_offset(const vector<size_t>& indices, const vector<size_t>& strides) {
    size_t offset = 0;
    for (int i = 0; i < indices.size(); i++) {
        offset += indices[i] * strides[i];
    }
    return offset;
}

// Function to advance a multi-dimensional index in a tensor. Returns true if the index was successfully incremented, or false if iteration is complete
bool Tensor::next_index(vector<size_t>& indices, const vector<size_t>& result_dims) {
    size_t num_dims = result_dims.size();
    for (size_t i = num_dims; i-- > 0;) {
        indices[i]++;
    if (indices[i] < result_dims[i]) {
            return true;
        } else {
            indices[i] = 0;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------

// Overload the + operator for element-wise addition between tensors
Tensor Tensor::operator+(Tensor& other) {
    Tensor result;

    // Need to perform broadcasting since the tensors have different shapes
    if (dimensions != other.dimensions) {
        vector<size_t> result_dims = broadcast_result_shape(dimensions, other.dimensions);
        result = Tensor(result_dims);

        vector<size_t> broadcasted_a_strides = broadcast_strides(dimensions, strides, result_dims);
        vector<size_t> broadcasted_b_strides = broadcast_strides(other.dimensions, other.strides, result_dims);
        vector<size_t> result_strides = compute_strides(result_dims);

        vector<size_t> indices(result_dims.size(), 0);

        do {
            size_t a_offset = compute_offset(indices, broadcasted_a_strides);
            size_t b_offset = compute_offset(indices, broadcasted_b_strides);
            size_t result_offset = compute_offset(indices, result_strides);

            // Add the values from both tensors and store them in the result tensor
            result.data.get()[result_offset] = data.get()[a_offset] + other.data.get()[b_offset];
        } while (next_index(indices, result_dims));
    }

    // Otherwise, the tensors have the same shape, so just add element-wise
    else {
        result = *this;
        for (size_t i = 0; i < other.total_elements; i++) {
            result.data.get()[i] += other.data.get()[i];
        }
    }

    if (requires_grad) {
        result.node = make_shared<AddBackward>(this, &other);
        result.node->tensor = &result;
    }

    return result;
}

// Overload the + operator for element-wise addition between tensors and scalars
Tensor Tensor::operator+(float value) {
    Tensor result = *this;
    for (size_t i = 0; i < total_elements; i++) {
        result.data.get()[i] += value;
    }
    return result;
}

// Overload the += operator for element-wise addition and assignment between tensors
Tensor& Tensor::operator+=(const Tensor& other) {
    for (size_t i = 0; i < other.total_elements; i++) {
        data.get()[i] += other.data.get()[i];
    }
    return *this;
}

// Overload the += operator for element-wise addition and assignment between tensors and scalars
Tensor& Tensor::operator+=(float value) {
    for (size_t i = 0; i < total_elements; i++) {
        data.get()[i] += value;
    }
    return *this;
}

// ---------------------------------------------------------------------------

// Overload the - operator for element-wise subtraction between tensors
Tensor Tensor::operator-(Tensor& other) {
    Tensor result;

    // Need to perform broadcasting since the tensors have different shapes
    if (dimensions != other.dimensions) {
        vector<size_t> result_dims = broadcast_result_shape(dimensions, other.dimensions);
        result = Tensor(result_dims);

        vector<size_t> broadcasted_a_strides = broadcast_strides(dimensions, strides, result_dims);
        vector<size_t> broadcasted_b_strides = broadcast_strides(other.dimensions, other.strides, result_dims);
        vector<size_t> result_strides = compute_strides(result_dims);

        vector<size_t> indices(result_dims.size(), 0);

        do {
            size_t a_offset = compute_offset(indices, broadcasted_a_strides);
            size_t b_offset = compute_offset(indices, broadcasted_b_strides);
            size_t result_offset = compute_offset(indices, result_strides);

            // Subtract the values from both tensors and store them in the result tensor
            result.data.get()[result_offset] = data.get()[a_offset] - other.data.get()[b_offset];
        } while (next_index(indices, result_dims));
    }

    // Otherwise, the tensors have the same shape, so just subtract element-wise
    else {
        result = *this;
        for (size_t i = 0; i < other.total_elements; i++) {
            result.data.get()[i] -= other.data.get()[i];
        }
    }

    if (requires_grad) {
        result.node = make_shared<SubBackward>(this, &other);
        result.node->tensor = &result;
    }

    return result;
}

// Overload the - operator for element-wise subtraction between tensors and scalars
Tensor Tensor::operator-(float value) {
    Tensor result = *this;
    for (size_t i = 0; i < total_elements; i++) {
        result.data.get()[i] -= value;
    }
    return result;
}

// Overload the -= operator for element-wise subtraction and assignment between tensors
Tensor& Tensor::operator-=(const Tensor& other) {
    for (size_t i = 0; i < other.total_elements; i++) {
        data.get()[i] -= other.data.get()[i];
    }
    return *this;
}

// Overload the -= operator for element-wise subtraction and assignment between tensors and scalars
Tensor& Tensor::operator-=(float value) {
    for (size_t i = 0; i < total_elements; i++) {
        data.get()[i] -= value;
    }
    return *this;
}

// ---------------------------------------------------------------------------

// Overload the * operator for element-wise multiplication between tensors
Tensor Tensor::operator*(Tensor& other) {
    Tensor result;

    // Need to perform broadcasting since the tensors have different shapes
    if (dimensions != other.dimensions) {
        vector<size_t> result_dims = broadcast_result_shape(dimensions, other.dimensions);
        result = Tensor(result_dims);

        vector<size_t> broadcasted_a_strides = broadcast_strides(dimensions, strides, result_dims);
        vector<size_t> broadcasted_b_strides = broadcast_strides(other.dimensions, other.strides, result_dims);
        vector<size_t> result_strides = compute_strides(result_dims);

        vector<size_t> indices(result_dims.size(), 0);

        do {
            size_t a_offset = compute_offset(indices, broadcasted_a_strides);
            size_t b_offset = compute_offset(indices, broadcasted_b_strides);
            size_t result_offset = compute_offset(indices, result_strides);

            // Multiply the values from both tensors and store them in the result tensor
            result.data.get()[result_offset] = data.get()[a_offset] * other.data.get()[b_offset];
        } while (next_index(indices, result_dims));
    }

    // Otherwise, the tensors have the same shape, so just multiply element-wise
    else {
        result = *this;
        for (size_t i = 0; i < other.total_elements; i++) {
            result.data.get()[i] *= other.data.get()[i];
        }
    }

    if (requires_grad) {
        result.node = make_shared<MulBackward>(this, &other);
        result.node->tensor = &result;
    }

    return result;
}

// Overload the * operator for element-wise multiplication between tensors and scalars
Tensor Tensor::operator*(float value) {
    Tensor result = *this;
    for (size_t i = 0; i < total_elements; i++) {
        result.data.get()[i] *= value;
    }
    return result;
}

// Overload the *= operator for element-wise multiplication and assignment between tensors
Tensor& Tensor::operator*=(const Tensor& other) {
    for (size_t i = 0; i < other.total_elements; i++) {
        data.get()[i] *= other.data.get()[i];
    }
    return *this;
}

// Overload the *= operator for element-wise multiplication and assignment between tensors and scalars
Tensor& Tensor::operator*=(float value) {
    for (size_t i = 0; i < total_elements; i++) {
        data.get()[i] *= value;
    }
    return *this;
}

// ---------------------------------------------------------------------------

// Overload the / operator for element-wise division between tensors
Tensor Tensor::operator/(Tensor& other) {
    Tensor result;

    // Need to perform broadcasting since the tensors have different shapes
    if (dimensions != other.dimensions) {
        vector<size_t> result_dims = broadcast_result_shape(dimensions, other.dimensions);
        result = Tensor(result_dims);

        vector<size_t> broadcasted_a_strides = broadcast_strides(dimensions, strides, result_dims);
        vector<size_t> broadcasted_b_strides = broadcast_strides(other.dimensions, other.strides, result_dims);
        vector<size_t> result_strides = compute_strides(result_dims);

        vector<size_t> indices(result_dims.size(), 0);

        do {
            size_t a_offset = compute_offset(indices, broadcasted_a_strides);
            size_t b_offset = compute_offset(indices, broadcasted_b_strides);
            size_t result_offset = compute_offset(indices, result_strides);

            // Divide the values from both tensors and store them in the result tensor
            result.data.get()[result_offset] = data.get()[a_offset] / other.data.get()[b_offset];
        } while (next_index(indices, result_dims));
    }

    // Otherwise, the tensors have the same shape, so just divide element-wise
    else {
        result = *this;
        for (size_t i = 0; i < other.total_elements; i++) {
            result.data.get()[i] /= other.data.get()[i];
        }
    }

    if (requires_grad) {
        result.node = make_shared<DivBackward>(this, &other);
        result.node->tensor = &result;
    }

    return result;
}

// Overload the / operator for element-wise division between tensors and scalars
Tensor Tensor::operator/(float value) {
    Tensor result = *this;
    for (size_t i = 0; i < total_elements; i++) {
        result.data.get()[i] /= value;
    }
    return result;
}

// Overload the /= operator for element-wise division and assignment between tensors
Tensor& Tensor::operator/=(const Tensor& other) {
    for (size_t i = 0; i < other.total_elements; i++) {
        data.get()[i] /= other.data.get()[i];
    }
    return *this;
}

// Overload the /= operator for element-wise division and assignment between tensors and scalars
Tensor& Tensor::operator/=(float value) {
    for (size_t i = 0; i < total_elements; i++) {
        data.get()[i] /= value;
    }
    return *this;
}

// ---------------------------------------------------------------------------

// Overload the << operator for printing the contents of a tensor
ostream& operator<<(ostream& os, const Tensor& tensor) {
    os << '(';
    for (size_t i = 0; i < tensor.total_elements; i++) {
        os << tensor.data.get()[i];
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

// Function to return the total number of elements in a tensor
size_t Tensor::numel() {
    return total_elements;
}

// Function to return the elements in a tensor
const float* Tensor::get_data() const {
    return data.get();
}

// ---------------------------------------------------------------------------

// Constructor for TensorSlice class used for chaining multiple [] operators
Tensor::TensorSlice::TensorSlice(shared_ptr<float> data, const vector<size_t>& dimensions,
    const vector<size_t>& strides, size_t offset, size_t level)
        : data(data),
          dimensions(dimensions),
          strides(strides),
          offset(offset),
          level(level) {}

// ---------------------------------------------------------------------------

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
    data.get()[offset] = value;
    return *this;
}

// Overload the float reference conversion operator to return data after the final []
Tensor::TensorSlice::operator float&() {
    if (level != dimensions.size()) {
        throw out_of_range("Cannot convert TensorSlice to float reference");
    }
    return data.get()[offset];
}

// ---------------------------------------------------------------------------

// Function to call Engine::run_backward() to compute the gradient of the current tensor w.r.t. graph leaves.
void Tensor::backward() {
    fill(grad.get(), grad.get() + total_elements, 1);
    Engine::run_backward(node);
}

// ---------------------------------------------------------------------------


int main() {
    Tensor a = Tensor::tensor({2, 4});
    cout << "The first element in the tensor a is " << a[0] << endl;
    cout << "The tensor a has the shape: " << a.shape_str() << endl << endl;

    vector<float> vec = {2, 4};
    Tensor b = Tensor::tensor(vec);
    cout << "The first element in the tensor b is " << b[0] << endl;
    cout << "The tensor b has the shape: " << b.shape_str() << endl << endl;

    Tensor c = Tensor::empty({2, 2});
    cout << "The first element in the tensor c is " << c[0][0] << endl;
    cout << "The tensor c has the shape: " << c.shape_str() << endl << endl;

    Tensor d = Tensor::zeros({2, 16});
    cout << "The first element in the tensor d is " << d[0][0] << endl;
    cout << "The tensor d has the shape: " << d.shape_str() << endl << endl;

    Tensor e = Tensor::ones({8, 4});
    cout << "The first element in the tensor e is " << e[0][0] << endl;
    cout << "The tensor e has the shape: " << e.shape_str() << endl << endl;

    cout << "The tensor e contains:" << endl;
    cout << e << endl;

    Tensor f = e + e;
    cout << "The sum of tensor e with itself is tensor f, which contains:" << endl;
    cout << f << endl << endl;

    Tensor g = e - f;
    cout << "The difference between tensor e and tensor f is tensor g, which contains:" << endl;
    cout << g << endl << endl;

    Tensor h = f * f;
    cout << "The product of tensor f with itself is tensor h, which contains:" << endl;
    cout << h << endl << endl;

    Tensor i = e / h;
    cout << "The quotient of tensor e and tensor h is tensor i, which contains:" << endl;
    cout << i << endl << endl;

    i[0][0] = 2;
    cout << "After assigning the value 2 to the first index in tensor i, it now contains:" << endl;
    cout << i << endl << endl;

    Tensor j; 
    j = Tensor::ones({4, 4});
    cout << "The tensor j created using the default constructor and move assignment operator contains:" << endl;
    cout << j << endl << endl;

    cout << "The product of tensor h with itself is a tensor containing:" << endl;
    cout << h * h << endl << endl;

    Tensor k = j + 2.5;
    cout << "The sum of tensor j and 2.5 is tensor k, which contains:" << endl;
    cout << k << endl << endl;

    Tensor l = j - 2.5;
    cout << "The difference between tensor j and 2.5 is tensor l, which contains:" << endl;
    cout << l << endl << endl;

    Tensor m = k * 2;
    cout << "The product of tensor k and 2 is tensor m, which contains:" << endl;
    cout << m << endl << endl;

    Tensor n = l / 2;
    cout << "The quotient of tensor l and 2 is tensor n, which contains:" << endl;
    cout << n << endl << endl;

    m += k;
    cout << "After adding tensor k to tensor m, tensor m now contains:" << endl;
    cout << m << endl << endl;

    m -= j;
    cout << "After subtracting tensor j from tensor m, tensor m now contains:" << endl;
    cout << m << endl << endl;

    m *= n;
    cout << "After multiplying tensor m by tensor n, tensor m now contains:" << endl;
    cout << m << endl << endl;

    m /= l;
    cout << "After dividing tensor m by tensor l, tensor m now contains:" << endl;
    cout << m << endl << endl;

    m += 1;
    cout << "After adding 1 to tensor m, it now contains:" << endl;
    cout << m << endl << endl;

    m -= 2.5;
    cout << "After subtracting 2.5 from tensor m, it now contains:" << endl;
    cout << m << endl << endl;

    m *= 2;
    cout << "After multiplying tensor m by 2, it now contains:" << endl;
    cout << m << endl << endl;

    m /= 3;
    cout << "After dividing tensor m by 3, it now contains:" << endl;
    cout << m << endl << endl;

    cout << "The tensor m has the shape: " << m.shape_str() << endl;
    Tensor o = m.view({2, 2, 4});
    cout << "The tensor m was reshaped with view() to a new tensor o with the shape: " << o.shape_str() << endl << endl;

    cout << "The tensor o has the shape: " << o.shape_str() << endl;
    Tensor p = o.view({2, 2, 2, 2});
    cout << "The tensor o was reshaped with view() to a new tensor p with the shape: " << p.shape_str() << endl << endl;

    cout << "The tensor p has the shape: " << p.shape_str() << endl;
    Tensor q = p.view({8, -1});
    cout << "The tensor p was reshaped with view() to a new tensor q with the shape: " << q.shape_str() << endl << endl;

    cout << "The tensor q has the shape: " << q.shape_str() << endl;
    q = q.flatten();
    cout << "The tensor q was flattened and now has the shape: " << q.shape_str() << endl << endl;

    cout << "The tensor o has the shape: " << o.shape_str() << endl;
    o = o.transpose(0, 2);
    cout << "The tensor o was transposed along dimensions 0 and 2 and now has the shape: " << o.shape_str() << endl << endl;

    cout << "The tensor j contains:" << endl << j << endl;
    j = j.sum();
    cout << "After applying sum to tensor j, it now contains:" << endl << j << endl << endl;

    cout << "The tensor f contains:" << endl << f << endl;
    f = f.pow(2);
    cout << "After applying pow(2) to tensor f, it now contains:" << endl << f << endl << endl;

    Tensor r = Tensor::tensor({1, 2});
    cout << "The tensor r contains:" << endl << r << endl;
    r = r.mean();
    cout << "After applying mean to tensor r, it now contains:" << endl << r << endl << endl;

    vector<size_t> vec2 = {4, 2};
    Tensor s = Tensor::empty(vec2);
    cout << "The tensor s has the shape: " << s.shape_str() << endl << endl;

    cout << "The tensor f has the shape: " << f.shape_str() << endl;
    cout << "The tensor f contains:" << endl << f << endl;
    Tensor t = Tensor::ones({4});
    cout << "The tensor t has the shape: " << t.shape_str() << endl;
    cout << "The tensor t contains:" << endl << t << endl;
    f = f + t;
    cout << "After adding tensor t to tensor f, tensor f now contains:" << endl << f << endl << endl;

    cout << "The tensor f contains:" << endl << f << endl;
    t *= 2;
    cout << "The tensor t contains:" << endl << t << endl;
    f = f - t;
    cout << "After subtracting tensor t from tensor f, tensor f now contains:" << endl << f << endl << endl;

    cout << "The tensor f contains:" << endl << f << endl;
    cout << "The tensor t contains:" << endl << t << endl;
    f = f * t;
    cout << "After multiplying tensor f by tensor t, tensor f now contains:" << endl << f << endl << endl;

    cout << "The tensor f contains:" << endl << f << endl;
    t += 1;
    cout << "The tensor t contains:" << endl << t << endl;
    f = f / t;
    cout << "After dividing tensor f by tensor t, tensor f now contains:" << endl << f << endl << endl;

    vector<size_t> vec3 = {4, 2};
    Tensor u = Tensor::zeros(vec2);
    cout << "The tensor u has the shape: " << u.shape_str() << endl << endl;

    vector<size_t> vec4 = {4, 2};
    Tensor v = Tensor::ones(vec2);
    cout << "The tensor v has the shape: " << v.shape_str() << endl << endl;

    Tensor w = Tensor::tensor({1, 2, 3, 4});
    cout << "The tensor w contains:" << endl << w << endl;
    Tensor x = w.min();
    cout << "The result of applying min() to tensor w is a new tensor x, which contains:" << endl << x << endl << endl;

    cout << "The tensor w contains:" << endl << w << endl;
    x = w.max();
    cout << "The result of applying max() to tensor w is a new tensor x, which contains:" << endl << x << endl << endl;
    
    cout << "The tensor w contains:" << endl << w << endl;
    bool equal = w.equal(w);
    cout << "The result of applying equal() to tensor w and itself is: " << equal << endl << endl;

    cout << "The tensor w contains:" << endl << w << endl;
    cout << "The tensor x contains:" << endl << x << endl;
    equal = w.equal(x);
    cout << "The result of applying equal() to tensor w and tensor x is: " << equal << endl << endl;

    Tensor y = Tensor::ones({4, 2, 4});
    y *= 3;
    cout << "The tensor y contains:" << endl << y << endl;
    Tensor z = Tensor::ones({4, 4, 2});
    z *= 2;
    cout << "The tensor z contains:" << endl << z << endl;    
    z = y.matmul(z);
    cout << "After applying matmul to tensors y and z and storing the result in z, the tensor z now contains:" << endl << z << endl;
    cout << "The tensor z has the shape: " << z.shape_str() << endl << endl;

    Tensor X = Tensor::ones({1, 784});
    X *= 0.5;
    cout << "The tensor X has the shape: " << X.shape_str() << endl;
    Tensor W1 = Tensor::ones({784, 256});
    W1 *= 0.01;
    X = X.matmul(W1);
    cout << "The tensor X has the shape: " << X.shape_str() << endl;
    Tensor W2 = Tensor::ones({256, 64});
    W2 *= 0.01;
    X = X.matmul(W2);
    cout << "The tensor X has the shape: " << X.shape_str() << endl;
    Tensor W3 = Tensor::ones({64, 10});
    W3 *= 0.01;
    X = X.matmul(W3);
    cout << "The tensor X has the shape: " << X.shape_str() << endl;
    cout << "The tensor X contains: " << X << endl << endl;

    Tensor A = Tensor::ones({2, 2});
    A *= 2;
    cout << "The tensor A contains:" << endl << A << endl;
    Tensor B = Tensor::ones({2, 2});
    cout << "The tensor B contains:" << endl << B << endl;
    Tensor C = A.matmul(B);
    cout << "The tensor C contains:" << endl << C << endl;
    Tensor D = Tensor::ones({2, 2});
    D *= 3;
    cout << "The tensor D contains:" << endl << D << endl;
    Tensor E = Tensor::ones({2, 2});
    E *= 4;
    cout << "The tensor E contains:" << endl << E << endl;
    Tensor F = D.matmul(E);
    cout << "The tensor F contains:" << endl << F << endl;

    Tensor G = C.matmul(F);
    cout << "The tensor G contains:" << endl << G << endl;
    cout << "The tensors A, B, C, D, E, and F which were subtracted to create the tensor G now have the gradients:" << endl;
    G.backward();
    cout << endl;

    return 0;
}
