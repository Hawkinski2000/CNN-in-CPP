#pragma once
#include "Tensor.h"
using namespace std;


class Optimizer {
    protected:
        vector<Tensor*> params;

    public:
        // Destructor for the Optimizer class
        virtual ~Optimizer();

        // Function to perform a single optimization step to update parameters
        virtual void step() = 0;

        // Function to perform a single optimization step to update parameters that runs on the GPU
        virtual void cuda_step() = 0;

        // Function to reset the gradients of all parameters
        void zero_grad();
};

// ---------------------------------------------------------------------------

// Stochastic Gradient Descent optimizer that inherits from the Optimizer class
class SGD : public Optimizer {
    float lr;

    public:
        // Constructor for the SGD class
        SGD(vector<Tensor*> params, float lr=0.001);

        // Function to perform a single optimization step to update parameters
        void step() override;

        // Function to perform a single optimization step to update parameters that runs on the GPU
        void cuda_step() override;
};
