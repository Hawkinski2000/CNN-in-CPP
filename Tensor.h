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

        // Function to create an empty tensor from a specified shape
        static Tensor empty(initializer_list<size_t> dims);

        // Function to create a tensor of zeros from a specified shape
        static Tensor zeros(initializer_list<size_t> dims);

        // Function to create a tensor of ones from a specified shape
        static Tensor ones(initializer_list<size_t> dims);

        // Function to create a tensor from specified values
        static Tensor tensor(initializer_list<float> values);

        // Function to create a tensor from a vector
        static Tensor tensor(vector<float>& values);

        // Overload the [] operator for indexing into the tensor data
        TensorSlice operator[](size_t index);

        // Overload the = operator to move a temporary tensor into an existing tensor
        Tensor& operator=(Tensor&& other);

        // Overload the = operator for copying one tensor to another
        Tensor& operator=(const Tensor& other);

        // Function to return a new tensor with the same data as the original tensor but of a different shape
        Tensor view(initializer_list<int> shape);

        // Function to flatten the input by reshaping it into a one-dimensional tensor
        Tensor flatten();

        // Overload the + operator for element-wise addition between tensors
        Tensor operator+(const Tensor& other);

        // Overload the + operator for element-wise addition between tensors and scalars
        Tensor operator+(float value);

        // Overload the += operator for element-wise addition and assignment between tensors
        Tensor& operator+=(const Tensor& other);

        // Overload the += operator for element-wise addition and assignment between tensors and scalars
        Tensor& operator+=(float value);

        // Overload the - operator for element-wise subtraction between tensors
        Tensor operator-(const Tensor& other);

        // Overload the - operator for element-wise subtraction between tensors and scalars
        Tensor operator-(float value);

        // Overload the -= operator for element-wise subtraction and assignment between tensors
        Tensor& operator-=(const Tensor& other);

        // Overload the -= operator for element-wise subtraction and assignment between tensors and scalars
        Tensor& operator-=(float value);

        // Overload the * operator for element-wise multiplication between tensors
        Tensor operator*(const Tensor& other);

        // Overload the * operator for element-wise multiplication between tensors and scalars
        Tensor operator*(float value);

        // Overload the *= operator for element-wise multiplication and assignment between tensors
        Tensor& operator*=(const Tensor& other);

        // Overload the *= operator for element-wise multiplication and assignment between tensors and scalars
        Tensor& operator*=(float value);

        // Overload the / operator for element-wise division between tensors
        Tensor operator/(const Tensor& other);

        // Overload the / operator for element-wise division between tensors and scalars
        Tensor operator/(float value);

        // Overload the /= operator for element-wise division and assignment between tensors
        Tensor& operator/=(const Tensor& other);

        // Overload the /= operator for element-wise division and assignment between tensors and scalars
        Tensor& operator/=(float value);

        // Overload the << operator for printing the contents of a tensor
        friend ostream& operator<<(ostream& os, const Tensor& tensor);

        // Function to return a tensor's shape vector
        const vector<size_t>& shape() const;

        // Function to return a tensor's shape as a string
        string shape_str();

    private:
        // Constructor for Tensor class used by creation methods that specify a shape
        Tensor(initializer_list<size_t> dims);

        // Constructor for Tensor class used by tensor() where values are specified
        Tensor(initializer_list<float> values);

        // Constructor for Tensor class used by tensor() to convert a vector to a tensor
        Tensor(const vector<float>& values);

        shared_ptr<vector<float>> data;
        vector<size_t> dimensions;
        vector<size_t> strides;
        size_t total_elements;
};
