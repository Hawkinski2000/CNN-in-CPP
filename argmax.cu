#include <iostream>
#include "Tensor.h"
#include "cuda_utils.cuh"


// Kernel for cuda_argmax() where a dimension to reduce was not specified, so all dimensions are reduced
__global__ void argmax_all_kernel(const float* __restrict__ data,
                                  int* __restrict__ result_index,
                                  float* __restrict__ result_value,
                                  size_t size) {
    extern __shared__ float sdata[]; // Array in shared memory for max values this block
    extern __shared__ int sindex[]; // Array in shared memory for max indices this block
    int tid = threadIdx.x; // Thread ID
    size_t i = blockIdx.x * blockDim.x + tid; // Index of element in data to process

    float val;
    int idx;
    if (i < size) {
        val = data[i];
        idx = static_cast<int>(i);
    }
    else {
        val = -INFINITY;
        idx = -1;
    }

    sdata[tid] = val;
    sindex[tid] = idx;
    __syncthreads();

    // Parallel reduction in shared memory. The range in data in the block to
    // find the max index in is halved each time until the max index of data
    // in the block is stored in sindex[0].
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) {
                sdata[tid] = sdata[tid + s];
                sindex[tid] = sindex[tid + s];
            }
        }
        __syncthreads();
    }

    // The max index of data in the block that was stored in sindex[0] will
    // replace result if it's the new overall max.
    if (tid == 0) {
        float prev_max = atomicMax((int*)result_value, __float_as_int(sdata[0]));
        if (__float_as_int(sdata[0]) > __float_as_int(prev_max)) {
            *result_index = sindex[0];
        }
    }
}

// Kernel for cuda_max() where a dimension to reduce was specified
__global__ void argmax_dim_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  const size_t* __restrict__ in_dims,
                                  const size_t* __restrict__ out_strides,
                                  size_t dim,
                                  size_t ndims,
                                  size_t total_elements) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_elements) {
        return;
    }

    extern __shared__ size_t shared_mem[]; // Array in shared memory for this block
    size_t* idx = &shared_mem[threadIdx.x * ndims]; // An array containing the multidimensional index of the output position to store into

    size_t idx_copy = i;
    for (int j = ndims - 1; j >= 0; --j) {
        idx[j] = idx_copy % in_dims[j];
        idx_copy /= in_dims[j];
    }

    idx[dim] = 0;

    size_t out_flat_idx = 0;
    for (size_t j = 0; j < ndims; j++) {
        out_flat_idx += idx[j] * out_strides[j];
    }

    float max_val = -INFINITY;
    int max_idx = 0;
    for (size_t k = 0; k < in_dims[dim]; k++) {
        idx[dim] = k;
        size_t in_flat_idx = 0;
        for (size_t j = 0; j < ndims; j++) {
            if (j == ndims - 1) {
                in_flat_idx += idx[j];
            }
            else {
                in_flat_idx += in_dims[j + 1];
            }
        }
        float cur_val = input[in_flat_idx];
        if (cur_val > max_val) {
            max_val = cur_val;
            max_idx = (int)k;
        }
    }

    // output[out_flat_idx] = max_idx;
    output[out_flat_idx] = static_cast<float>(max_idx);
}

// Function to return the indices of the maximum value of all elements in a tensor that runs on the GPU
Tensor Tensor::cuda_argmax(optional<size_t> dim) {
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
        argmax_dim_kernel<<<blocks, threads, shared_mem_size>>>(data.get(),
                                                                // reinterpret_cast<int*>(result.data.get()),
                                                                result.data.get(),
                                                                d_in_dims,
                                                                d_out_strides,
                                                                d,
                                                                ndims,
                                                                total_elements);

        cudaDeviceSynchronize();

        cudaFree(d_in_dims);
        cudaFree(d_out_strides);
    } else {
        // A dimension to reduce was not specified, so all dimensions are reduced
        float* d_max_value;
        // int* d_max_index;
        float* d_max_index;

        cudaMalloc(&d_max_value, sizeof(float));
        // cudaMalloc(&d_max_index, sizeof(int));
        cudaMalloc(&d_max_index, sizeof(float));
        cuda_fill(d_max_value, 1, -INFINITY);

        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;
        size_t shared_mem = threads * sizeof(float);
        argmax_all_kernel<<<blocks, threads, shared_mem>>>(data.get(),
                                                        //    d_max_index,
                                                           reinterpret_cast<int*>(d_max_index),
                                                           d_max_value,
                                                           total_elements);

        cudaDeviceSynchronize();

        result = Tensor::empty({1}, true);
        // result.data = shared_ptr<float>(reinterpret_cast<float*>(d_max_index), [](float* p) { cudaFree(p); });
        result.data = shared_ptr<float>(d_max_index, [](float* p) { cudaFree(p); });

        cudaFree(d_max_value);
    }

    return result;
}
