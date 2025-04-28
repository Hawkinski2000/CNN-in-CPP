#pragma once

#include <iostream>
#include <vector>
using namespace std;


class Tensor {
    public:
        Tensor(initializer_list<int> dims);
        
        vector<int> shape();

    private:
        vector<int> dimensions;
    
};