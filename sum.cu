#include <iostream>
#include "Tensor.h"


// Kernel for cuda_sum() where a dimension to reduce was not specified, so all dimensions are reduced
__global__ void sum_all_kernel(const float* __restrict__ data, float* result, size_t size) {
    extern __shared__ float sdata[]; // array in shared memory for this block
    int tid = threadIdx.x; // Thread ID
    size_t i = blockIdx.x * blockDim.x + threadIdx.x; // Index of element in data to process

    float val;
    if (i < size) {
        val = data[i];
    }
    else {
        val = 0.0f;
    }
    sdata[tid] = val;
    __syncthreads();

    // Parallel reduction in shared memory. The range in data to sum is
    // halved each time until the sum of data in the block is accumulated in
    // sdata[0]. E.g., if data contains [1, 2, 3, 4], after each iteration,
    // sdata will contain [1+3, 2+4, 3, 4], then [4+6, 6, 3, 4], where
    // sdata[0] = 10, the sum of [1, 2, 3, 4].  
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // The sum of data in the block that was accumulated in sdata[0] will be
    // added to result.
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

// Kernel for cuda_sum() where a dimension to reduce was specified
__global__ void sum_dim_kernel(const float* __restrict__ input,
                               float* output,
                               const size_t* in_dims,
                               const size_t* out_strides,
                               size_t dim,
                               size_t ndims,
                               size_t total_elements) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_elements) {
        return;
    }

    extern __shared__ size_t shared_mem[]; // array in shared memory for this block
    size_t* idx = &shared_mem[threadIdx.x * ndims]; // An array containing the multidimensional index of the output position to accumulate into

    size_t idx_copy = i;
    for (size_t j = ndims; j-- > 0;) {
        idx[j] = idx_copy % in_dims[j];
        idx_copy /= in_dims[j];
    }

    idx[dim] = 0;

    size_t out_flat_idx = 0;
    for (size_t j = 0; j < ndims; j++) {
        out_flat_idx += idx[j] * out_strides[j];
    }

    // The sum of data in the block that was accumulated in sdata[0] will be
    // added to result.
    atomicAdd(&output[out_flat_idx], input[i]);
}


// Function to return the sum of all elements in a tensor that runs on the GPU
Tensor Tensor::cuda_sum(optional<size_t> dim) {
    Tensor result;

    // A dimension to reduce was specified
    if (dim.has_value()) {
        size_t d = dim.value();
        vector<size_t> out_dims = dimensions;
        out_dims[d] = 1;

        result = Tensor::zeros(out_dims, true);

        size_t ndims = dimensions.size();

        // Allocate device memory for in_dims and out_strides
        size_t* d_in_dims;
        size_t* d_out_strides;
        cudaMalloc(&d_in_dims, ndims * sizeof(size_t));
        cudaMemcpy(d_in_dims, dimensions.data(), ndims * sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMalloc(&d_out_strides, ndims * sizeof(size_t));
        cudaMemcpy(d_out_strides, result.strides.data(), ndims * sizeof(size_t), cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        size_t shared_mem_size = threads * ndims * sizeof(size_t);
        sum_dim_kernel<<<blocks, threads, shared_mem_size>>>(data.get(),
                                                             result.data.get(),
                                                             d_in_dims,
                                                             d_out_strides,
                                                             d,
                                                             ndims,
                                                             total_elements);

        cudaDeviceSynchronize();

        cudaFree(d_in_dims);
        cudaFree(d_out_strides);
    }
    else {
        // A dimension to reduce was not specified, so all dimensions are reduced
        float* d_result;
        
        cudaMalloc(&d_result, sizeof(float));
        cudaMemset(d_result, 0, sizeof(float));

        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        size_t shared_mem_size = threads * sizeof(float);
        sum_all_kernel<<<blocks, threads, shared_mem_size>>>(data.get(), d_result, total_elements);

        result = Tensor::empty({1}, true);
        result.data = shared_ptr<float>(d_result, [](float* p) { cudaFree(p); });
    }
    return result;
}
