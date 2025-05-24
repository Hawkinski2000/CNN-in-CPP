#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "Tensor.h"


/*
==============================================================================
This function uses code from Simon Boehm's repository, "SGEMM_CUDA":
    https://github.com/siboehm/SGEMM_CUDA/tree/master
==============================================================================
*/

Tensor Tensor::matmul(Tensor& other) {
    cudaSetDevice(0);
    
    cublasHandle_t handle;
    cublasCreate(&handle);

    if (dimensions.size() == 1) {
        dimensions = {1, dimensions[0]};
    }
    if (other.dimensions.size() == 1) {
        other.dimensions = {1, other.dimensions[0]};
    }

    size_t m, n, k;
    m = dimensions[dimensions.size() - 2];
    k = dimensions[dimensions.size() - 1];
    n = other.dimensions[other.dimensions.size() - 1];

    vector<size_t> A_batch_dims;
    if (dimensions.size() > 2) {
        A_batch_dims = vector<size_t>(dimensions.begin(), dimensions.end() - 2);
    } else {
        A_batch_dims = {1};
    }
    vector<size_t> B_batch_dims;
    if (other.dimensions.size() > 2) {
        B_batch_dims = vector<size_t>(other.dimensions.begin(), other.dimensions.end() - 2);
    } else {
        B_batch_dims = {1};
    }

    size_t A_batch_count = 1;
    size_t B_batch_count = 1;
    for (size_t dim : A_batch_dims) {
        A_batch_count *= dim;
    }
    for (size_t dim : B_batch_dims) {
        B_batch_count *= dim;
    }

    size_t strideA = m * k, strideB = k * n, strideC = m * n;
    if (A_batch_dims.size() == 1 && A_batch_dims[0] == 1) {
        strideA = 0;
    }
    else if (B_batch_dims.size() == 1 && B_batch_dims[0] == 1) {
        strideB = 0;
    }
    
    size_t batch_count = 1;
    size_t A_copies = 1;
    size_t B_copies = 1;
    vector<size_t> result_batch_dims;
    if (A_batch_dims != B_batch_dims) {
        if (strideA == 0) {
            batch_count = B_batch_count;
            result_batch_dims = B_batch_dims;
        }
        else if (strideB == 0) {
            batch_count = A_batch_count;
            result_batch_dims = A_batch_dims;
        }
        else {
            result_batch_dims = broadcast_result_shape(A_batch_dims, B_batch_dims);
            for (size_t dim : result_batch_dims) {
                batch_count *= dim;
            }
            A_copies = batch_count / A_batch_count;
            B_copies = batch_count / B_batch_count;
        }
    }
    else {
        batch_count = A_batch_count;
    }

    float *dA, *dB, *dC;

    float* A;
    float* B;
    shared_ptr<float> A_expanded;
    shared_ptr<float> B_expanded;
    if (A_copies > 1) {
        A_expanded = shared_ptr<float>(new float[A_copies * total_elements], default_delete<float[]>());
        for (size_t i = 0; i < A_copies; i++) {
            copy(data.get(), data.get() + total_elements, A_expanded.get() + i * total_elements);
        }
        A = A_expanded.get();
    }
    else {
        A = data.get();
    }
    if (B_copies > 1) {
        B_expanded = shared_ptr<float>(new float[B_copies * other.total_elements], default_delete<float[]>());
        for (size_t i = 0; i < B_copies; i++) {
            copy(other.data.get(), other.data.get() + other.total_elements, B_expanded.get() + i * other.total_elements);
        }
        B = B_expanded.get();
    }
    else {
        B = other.data.get();
    }
    float* C = new float[batch_count * m * n];

    size_t A_mem = sizeof(float) * m * k;
    size_t B_mem = sizeof(float) * k * n;
    size_t C_mem = sizeof(float) * batch_count * m * n;
    if (strideA > 0) {
        A_mem *= batch_count;
    }
    if (strideB > 0) {
        B_mem *= batch_count;
    }

    cudaMalloc((void **)&dA, A_mem);
    cudaMalloc((void **)&dB, B_mem);
    cudaMalloc((void **)&dC, C_mem);

    cudaMemcpy(dA, A, A_mem, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, B_mem, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C, C_mem, cudaMemcpyHostToDevice);

    float alpha = 1, beta = 0; // GEMM input parameters, C=α*AB+β*C
    
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dA, CUDA_R_32F,
                m, strideA, dB, CUDA_R_32F, k, strideB, &beta, dC, CUDA_R_32F, m, strideC, batch_count,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    cudaDeviceSynchronize();
    cudaMemcpy(C, dC, C_mem, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cublasDestroy(handle);

    Tensor result;
    if (batch_count > 1) {
        vector<size_t> result_dims(result_batch_dims.size() + 2);
        copy(result_batch_dims.begin(), result_batch_dims.end(), result_dims.begin());
        result_dims[result_batch_dims.size()] = m;
        result_dims[result_batch_dims.size() + 1] = n;
        result = Tensor::empty(result_dims);
    }
    else {
        result = Tensor::empty({m, n});
    }

    result.data = shared_ptr<float>(C, default_delete<float[]>());
    
    if (requires_grad) {
        result.node = make_shared<MatmulBackward>(this, &other);
        result.node->tensor = &result;
    }

    return result;
}
