#include <iostream>
#include "Tensor.h"
using namespace std;


Tensor::Tensor(initializer_list<int> dims) {
    for (int i : dims) {
        dimensions.push_back(i);
    }
}

vector<int> Tensor::shape() {
    return dimensions;
}

int main() {
    Tensor t({2, 4, 8});

    vector<int> shape = t.shape();

    cout << '(';
    for (size_t i = 0; i < shape.size(); i++) {
        cout << shape[i];
        if (i != shape.size() - 1) {
            cout << ", ";
        } 
    }
    cout << ')' << endl;

    return 0;
}
