#pragma once

#include <iostream>
#include <vector>
using namespace std;


class Tensor {
    public:
        class TensorSlice {
            public:
                // Constructor for TensorSlice class used for chaining multiple [] operators
                TensorSlice(float* data, const vector<size_t>& dimensions,
                    const vector<size_t>& strides, size_t offset, size_t level);

                // Overload the [] operator for indexing into the tensor data
                TensorSlice operator[](size_t index);

                // Overload the float reference conversion operator to return data after the final []
                operator float&();
        
            private:
                float* data = nullptr;
                vector<size_t> dimensions;
                vector<size_t> strides;
                size_t offset;
                size_t level;
        };

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

        // Overload the << operator for printing the contents of a tensor
        friend ostream& operator<<(ostream& os, Tensor& tensor);

        // Function to return a tensor's shape vector
        const vector<size_t>& shape() const;

        // Function to return a tensor's shape as a string
        string shape_str();

        // Destructor for Tensor class
        ~Tensor();

    private:
        // Constructor for Tensor class used by creation methods that specify a shape
        Tensor(initializer_list<size_t> dims);

        // Constructor for Tensor class used by tensor() where values are specified
        Tensor(initializer_list<float> values);

        // Constructor for Tensor class used by tensor() to convert a vector to a tensor
        Tensor(vector<float>& values);

        float* data = nullptr;
        vector<size_t> dimensions;
        vector<size_t> strides;
        size_t total_elements = 0;
};
