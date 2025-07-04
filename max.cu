#include <iostream>
#include "Tensor.h"
#include "cuda_utils.cuh"


// Kernel for cuda_max() where a dimension to reduce was not specified, so all dimensions are reduced
__global__ void max_all_kernel(const float* __restrict__ data, float* result, size_t size) {
    extern __shared__ float sdata[]; // Array in shared memory for this block
    int tid = threadIdx.x; // Thread ID
    size_t i = blockIdx.x * blockDim.x + threadIdx.x; // Index of element in data to process

    float val;
    if (i < size) {
        val = data[i];
    }
    else {
        val = -INFINITY;
    }
    sdata[tid] = val;
    __syncthreads();

    // Parallel reduction in shared memory. The range in data in the block to
    // find the max in is halved each time until the max of data in the block
    // is stored in sdata[0]. E.g., if data contains [1, 2, 3, 4], after
    // each iteration, sdata will contain [3, 4, 3, 4], then [4, 4, 3, 4],
    // where sdata[0] = 4, the max of [1, 2, 3, 4].
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // The max of data in the block that was stored in sdata[0] will replace
    // result if it's the new overall max.
    if (tid == 0) {
        atomicMax((int*)result, __float_as_int(sdata[0]));
    }
}

// Kernel for cuda_max() where a dimension to reduce was specified
__global__ void max_dim_kernel(const float* __restrict__ input,
                               float* output,
                               const size_t* in_dims,
                               const size_t* out_strides,
                               size_t dim,
                               size_t ndims,
                               size_t total_elements) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x; // Index of element in data to process
    if (i >= total_elements) {
        return;
    }

    extern __shared__ size_t shared_mem[]; // Array in shared memory for this block
    size_t* idx = &shared_mem[threadIdx.x * ndims]; // An array containing the multidimensional index of the output position to store into

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

    // The max of data in the block that was stored in sdata[0] will replace
    // result if it's the new overall max.
    atomicMax((int*)&output[out_flat_idx], __float_as_int(input[i]));
}

// Function to return the maximum value of all elements in a tensor that runs on the GPU
Tensor Tensor::cuda_max(optional<size_t> dim) {
    Tensor result;

    // A dimension to reduce was specified
    if (dim.has_value()) {
        size_t d = dim.value();
        vector<size_t> out_dims = dimensions;
        out_dims[d] = 1;

        result = Tensor::empty(out_dims, true);
        cuda_fill(result.data.get(), result.total_elements, -INFINITY);

        size_t ndims = dimensions.size();

        // Allocate device memory for in_dims and out_strides
        size_t* d_in_dims;
        size_t* d_out_strides;
        cudaMalloc(&d_in_dims, ndims * sizeof(size_t));
        cudaMemcpy(d_in_dims, dimensions.data(), ndims * sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMalloc(&d_out_strides, ndims * sizeof(size_t));
        cudaMemcpy(d_out_strides, result.strides.data(), ndims * sizeof(size_t), cudaMemcpyHostToDevice);

        int threads = 1024;
        int blocks = (total_elements + threads - 1) / threads;
        size_t shared_mem_size = threads * ndims * sizeof(size_t);
        max_dim_kernel<<<blocks, threads, shared_mem_size>>>(data.get(),
                                                             result.data.get(),
                                                             d_in_dims,
                                                             d_out_strides,
                                                             d,
                                                             ndims,
                                                             total_elements);
                                                             
        cudaFree(d_in_dims);
        cudaFree(d_out_strides);
    }
    else {
        // A dimension to reduce was not specified, so all dimensions are reduced
        float* d_result;

        cudaMalloc(&d_result, sizeof(float));
        cuda_fill(d_result, 1, -INFINITY);

        int threads = 1024;
        int blocks = (total_elements + threads - 1) / threads;
        size_t shared_mem_size = threads * sizeof(float);
        max_all_kernel<<<blocks, threads, shared_mem_size>>>(data.get(), d_result, total_elements);

        result = Tensor::empty({1}, true);
        result.data = shared_ptr<float>(d_result, [](float* p) { cudaFree(p); });
    }
    return result;
}
