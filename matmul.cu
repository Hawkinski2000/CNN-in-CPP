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

Tensor Tensor::matmul(const Tensor& other) {
    cudaSetDevice(0);
    
    cublasHandle_t handle;
    cublasCreate(&handle);

    size_t m, n, k;
    m = dimensions[dimensions.size() - 2];
    k = dimensions[dimensions.size() - 1];
    n = other.dimensions[other.dimensions.size() - 1];

    vector<size_t> batch_dims;
    if (dimensions.size() > 2) {
        batch_dims = vector<size_t>(dimensions.begin(), dimensions.end() - 2);
    } else {
        batch_dims = {1};
    }

    size_t batch_count = 1;
    for (size_t dim : batch_dims) {
        batch_count *= dim;
    }

    float alpha = 1, beta = 0; // GEMM input parameters, C=α*AB+β*C

    float *dA, *dB, *dC;

    float* A = data.get();
    float* B = other.data.get();
    float* C = new float[batch_count * m * n];

    cudaMalloc((void **)&dA, sizeof(float) * batch_count * m * k);
    cudaMalloc((void **)&dB, sizeof(float) * batch_count * k * n);
    cudaMalloc((void **)&dC, sizeof(float) * batch_count * m * n);

    cudaMemcpy(dA, A, sizeof(float) * batch_count * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(float) * batch_count * k * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C, sizeof(float) * batch_count * m * n, cudaMemcpyHostToDevice);

    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, dB, CUDA_R_32F,
                n, (k * n), dA, CUDA_R_32F, k, (m * k), &beta, dC, CUDA_R_32F, n, (m * n), batch_count,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    cudaDeviceSynchronize();
    cudaMemcpy(C, dC, sizeof(float) * batch_count * m * n, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cublasDestroy(handle);

    Tensor result;
    if (batch_count > 1) {
        result = Tensor::empty({batch_count, m, n});
    }
    else {
        result = Tensor::empty({m, n});
    }
    result.data = shared_ptr<float>(C, default_delete<float[]>());

    return result;
}
