#include <iostream>
#include "Tensor.h"


// Kernel for cuda_add() for when broadcasting is not needed
__global__ void add_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, size_t total_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        C[idx] = A[idx] + B[idx];
    }
}

// Kernel for cuda_add() for when broadcasting is needed
__global__ void broadcast_add_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     const size_t* __restrict__ dims,
                                     const size_t* __restrict__ a_strides,
                                     const size_t* __restrict__ b_strides,
                                     size_t ndim,
                                     size_t total_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) {
        return;
    }

    size_t a_offset = 0, b_offset = 0;
    size_t linear_idx = idx;

    for (size_t dim = ndim; dim-- > 0;) {
        size_t dim_idx = linear_idx % dims[dim];
        linear_idx /= dims[dim];

        if (a_strides[dim] > 0) {
            a_offset += dim_idx * a_strides[dim];
        }
        if (b_strides[dim] > 0) {
            b_offset += dim_idx * b_strides[dim];
        }
    }

    C[idx] = A[a_offset] + B[b_offset];
}

// Function for element-wise addition between tensors that runs on the GPU
Tensor Tensor::cuda_add(Tensor& other) {
    Tensor result;

    // Need to perform broadcasting since the tensors have different shapes
    if (dimensions != other.dimensions) {
        vector<size_t> result_dims = broadcast_result_shape(dimensions, other.dimensions);
        result = Tensor(result_dims, true);

        vector<size_t> a_strides = broadcast_strides(dimensions, strides, result_dims);
        vector<size_t> b_strides = broadcast_strides(other.dimensions, other.strides, result_dims);
        vector<size_t> result_strides = compute_strides(result_dims);

        size_t* d_result_dims;
        size_t* d_a_strides;
        size_t* d_b_strides;
        cudaMalloc(&d_result_dims, result_dims.size() * sizeof(size_t));
        cudaMalloc(&d_a_strides, result_dims.size() * sizeof(size_t));
        cudaMalloc(&d_b_strides, result_dims.size() * sizeof(size_t));
        cudaMemcpy(d_result_dims, result_dims.data(), result_dims.size() * sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_a_strides, a_strides.data(), result_dims.size() * sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b_strides, b_strides.data(), result_dims.size() * sizeof(size_t), cudaMemcpyHostToDevice);

        int threads = 1024;
        int blocks = (result.total_elements + threads - 1) / threads;
        broadcast_add_kernel<<<blocks, threads>>>(this->data.get(),
                                                  other.data.get(),
                                                  result.data.get(),
                                                  d_result_dims,
                                                  d_a_strides,
                                                  d_b_strides,
                                                  result_dims.size(),
                                                  result.total_elements);

        

        cudaFree(d_result_dims);
        cudaFree(d_a_strides);
        cudaFree(d_b_strides);
    }

    // Otherwise, the tensors have the same shape, so just add element-wise
    else {
        result = Tensor(this->dimensions, true);

        int threads = 1024;
        int blocks = (total_elements + threads - 1) / threads;
        add_kernel<<<blocks, threads>>>(this->data.get(), other.data.get(), result.data.get(), result.total_elements);

        
    }

    return result;
}

// ---------------------------------------------------------------------------

// Kernel for cuda_add_scalar()
__global__ void add_scalar_kernel(float* __restrict__ A, float* __restrict__ B, float value, size_t total_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        B[idx] = A[idx] + value;
    }
}

// Function for element-wise addition between tensors and scalars that runs on the GPU
Tensor Tensor::cuda_add_scalar(float value) {
    Tensor result(this->dimensions, true);
    int threads = 1024;
    int blocks = (total_elements + threads - 1) / threads;
    add_scalar_kernel<<<blocks, threads>>>(this->data.get(), result.data.get(), value, this->total_elements);
    
    return result;
}

// ---------------------------------------------------------------------------

// Kernel for cuda_add_()
__global__ void add_inplace_kernel(float* A, float* B, size_t total_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        A[idx] += B[idx];
    }
}

// Function for element-wise addition and assignment between tensors that runs on the GPU
Tensor& Tensor::cuda_add_(const Tensor& other) {
    int threads = 1024;
    int blocks = (total_elements + threads - 1) / threads;
    add_inplace_kernel<<<blocks, threads>>>(this->data.get(), other.data.get(), this->total_elements);
    
    return *this;
}

// ---------------------------------------------------------------------------

// Kernel for cuda_add_scalar_()
__global__ void add_scalar_inplace_kernel(float* A, float value, size_t total_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        A[idx] += value;
    }
}

// Function for element-wise addition and assignment between tensors and scalars that runs on the GPU
Tensor& Tensor::cuda_add_scalar_(float value) {
    int threads = 1024;
    int blocks = (total_elements + threads - 1) / threads;
    add_scalar_inplace_kernel<<<blocks, threads>>>(this->data.get(), value, this->total_elements);
    
    return *this;
}
