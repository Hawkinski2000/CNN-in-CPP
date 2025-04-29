#pragma once

#include <iostream>
#include <vector>
using namespace std;


class Tensor {
    public:
        // Constructor for Tensor class
        Tensor(initializer_list<size_t> dims);

        // Destructor for Tensor class
        ~Tensor();
        
        // Function to return a tensor's shape vector
        const vector<size_t>& shape() const;

        // Function to return a tensor's shape as a string
        string shape_str();

    private:
        float* data = nullptr;
        vector<size_t> dimensions;
        vector<size_t> strides;
        int total_elements = 0;
};
