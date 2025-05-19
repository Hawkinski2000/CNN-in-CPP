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
    m = dimensions[0];
    k = dimensions[1];
    n = other.dimensions[1];

    float alpha = 1, beta = 0; // GEMM input parameters, C=α*AB+β*C

    float *dA, *dB, *dC;

    float* A = data.get();
    float* B = other.data.get();
    float* C = new float[m * n];

    cudaMalloc((void **)&dA, sizeof(float) * m * k);
    cudaMalloc((void **)&dB, sizeof(float) * k * n);
    cudaMalloc((void **)&dC, sizeof(float) * m * n);

    cudaMemcpy(dA, A, sizeof(float) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(float) * k * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C, sizeof(float) * m * n, cudaMemcpyHostToDevice);

    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, dB, CUDA_R_32F,
                n, dA, CUDA_R_32F, k, &beta, dC, CUDA_R_32F, n, CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    cudaDeviceSynchronize();
    cudaMemcpy(C, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cublasDestroy(handle);

    Tensor result = Tensor::empty({m, n});
    result.data = shared_ptr<float>(C, default_delete<float[]>());

    return result;
}
