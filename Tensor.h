#pragma once

#include <iostream>
#include <vector>
using namespace std;


class Tensor {
    public:
        Tensor(initializer_list<int> dims);
        
        const vector<int>& shape() const;

        string shape_str();

    private:
        vector<int> dimensions;
};