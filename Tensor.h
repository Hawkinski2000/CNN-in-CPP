#pragma once

#include <iostream>
#include <vector>
#include <memory>
using namespace std;


class Tensor {
    public:
        class TensorSlice {
            public:
                // Constructor for TensorSlice class used for chaining multiple [] operators
                TensorSlice(shared_ptr<vector<float>> data, const vector<size_t>& dimensions,
                    const vector<size_t>& strides, size_t offset, size_t level);

                // ---------------------------------------------------------------------------

                // Overload the [] operator for indexing into the tensor data
                TensorSlice operator[](size_t index);

                // Overload the = operator for assigning values to indices in the tensor
                TensorSlice& operator=(float value);

                // Overload the float reference conversion operator to return data after the final []
                operator float&();
        
            private:
                shared_ptr<vector<float>> data;
                vector<size_t> dimensions;
                vector<size_t> strides;
                size_t offset;
                size_t level;
        };

        // Default constructor for the Tensor class
        Tensor();

        // Copy constructor for the Tensor class
        Tensor(const Tensor& other);

        // Move constructor for the Tensor class
        Tensor(Tensor&& other);

        // ---------------------------------------------------------------------------

        // Function to create an empty tensor from a shape specified as an initializer_list
        static Tensor empty(initializer_list<size_t> dims);

        // Function to create an empty tensor from a shape specified as a vector
        static Tensor empty(vector<size_t> dims);

        // Function to create a tensor of zeros from a specified shape
        static Tensor zeros(initializer_list<size_t> dims);

        // Function to create tensor of zeros from a shape specified as a vector
        static Tensor zeros(vector<size_t> dims);

        // Function to create a tensor of ones from a specified shape
        static Tensor ones(initializer_list<size_t> dims);

        // Function to create tensor of ones from a shape specified as a vector
        static Tensor ones(vector<size_t> dims);

        // Function to create a tensor from specified values
        static Tensor tensor(initializer_list<float> values);

        // Function to create a tensor from a vector
        static Tensor tensor(vector<float>& values);

        // ---------------------------------------------------------------------------

        // Overload the [] operator for indexing into the tensor data
        TensorSlice operator[](size_t index);

        // Overload the = operator to move a temporary tensor into an existing tensor
        Tensor& operator=(Tensor&& other);

        // Overload the = operator for copying one tensor to another
        Tensor& operator=(const Tensor& other);

        // ---------------------------------------------------------------------------

        // Function to return a new tensor with the same data as the original tensor but of a different shape
        Tensor view(initializer_list<int> shape);

        // Function to flatten a tensor by reshaping it into a one-dimensional tensor
        Tensor flatten();

        // Function to return a tensor that is a transposed version of a tensor
        Tensor transpose(size_t dim0, size_t dim1);

        // ---------------------------------------------------------------------------

        // Function to return the sum of all elements in a tensor
        Tensor sum();

        // Function to return the mean of all elements in a tensor
        Tensor mean();

        // Function for element-wise powers between tensors and scalars
        Tensor pow(int exponent);

        // Function to return the minimum value of all elements in a tensor
        Tensor min();

        // Function to return the maximum value of all elements in a tensor
        Tensor max();

        // Function to return true if two tensors have the same shape and elements, otherwise false.
        bool equal(const Tensor& other);

        // Function to return the matrix product of two tensors.
        Tensor matmul(Tensor& other);

        // ---------------------------------------------------------------------------

        // Function to compute a tensor's strides from its dimensions
        vector<size_t> compute_strides(const vector<size_t>& dimensions);

        // Function to pad a shape or strides vector with 1's or 0's for broadcasting
        vector<size_t> pad_vector(const vector<size_t>& original, size_t num_dims, size_t pad_with);

        // Function to return a vector containing the resulting tensor shape from an operation on two broadcastable tensors
        vector<size_t> broadcast_result_shape(const vector<size_t>& a_dims, const vector<size_t>& b_dims);

        // Function to return a vector containing the broadcastable strides of a tensor given a resulting tensor shape
        vector<size_t> broadcast_strides(const vector<size_t>& original_dims, const vector<size_t>& original_strides, const vector<size_t>& result_dims);

        // Function to compute the memory offset for a multi-dimensional index from the tensor's strides
        size_t compute_offset(const vector<size_t>& indices, const vector<size_t>& strides);

        // Function to advance a multi-dimensional index in a tensor. Returns true if the index was successfully incremented, or false if iteration is complete
        bool next_index(vector<size_t>& indices, const vector<size_t>& result_dims);

        // ---------------------------------------------------------------------------

        // Overload the + operator for element-wise addition between tensors
        Tensor operator+(const Tensor& other);

        // Overload the + operator for element-wise addition between tensors and scalars
        Tensor operator+(float value);

        // Overload the += operator for element-wise addition and assignment between tensors
        Tensor& operator+=(const Tensor& other);

        // Overload the += operator for element-wise addition and assignment between tensors and scalars
        Tensor& operator+=(float value);

        // ---------------------------------------------------------------------------

        // Overload the - operator for element-wise subtraction between tensors
        Tensor operator-(const Tensor& other);

        // Overload the - operator for element-wise subtraction between tensors and scalars
        Tensor operator-(float value);

        // Overload the -= operator for element-wise subtraction and assignment between tensors
        Tensor& operator-=(const Tensor& other);

        // Overload the -= operator for element-wise subtraction and assignment between tensors and scalars
        Tensor& operator-=(float value);

        // ---------------------------------------------------------------------------

        // Overload the * operator for element-wise multiplication between tensors
        Tensor operator*(const Tensor& other);

        // Overload the * operator for element-wise multiplication between tensors and scalars
        Tensor operator*(float value);

        // Overload the *= operator for element-wise multiplication and assignment between tensors
        Tensor& operator*=(const Tensor& other);

        // Overload the *= operator for element-wise multiplication and assignment between tensors and scalars
        Tensor& operator*=(float value);

        // ---------------------------------------------------------------------------

        // Overload the / operator for element-wise division between tensors
        Tensor operator/(const Tensor& other);

        // Overload the / operator for element-wise division between tensors and scalars
        Tensor operator/(float value);

        // Overload the /= operator for element-wise division and assignment between tensors
        Tensor& operator/=(const Tensor& other);

        // Overload the /= operator for element-wise division and assignment between tensors and scalars
        Tensor& operator/=(float value);

        // ---------------------------------------------------------------------------

        // Overload the << operator for printing the contents of a tensor
        friend ostream& operator<<(ostream& os, const Tensor& tensor);

        // Function to return a tensor's shape vector
        const vector<size_t>& shape() const;

        // Function to return a tensor's shape as a string
        string shape_str();

    private:
        // Constructor for Tensor class used by creation methods that specify a shape as an initializer_list
        Tensor(initializer_list<size_t> dims);

        // Constructor for Tensor class used by creation methods that specify a shape as a vector
        Tensor(const vector<size_t>& dims);

        // Constructor for Tensor class used by tensor() where values are specified
        Tensor(initializer_list<float> values);

        // Constructor for Tensor class used by tensor() to convert a vector to a tensor
        Tensor(const vector<float>& values);

        // ---------------------------------------------------------------------------

        shared_ptr<vector<float>> data;
        vector<size_t> dimensions;
        vector<size_t> strides;
        size_t total_elements;
};
