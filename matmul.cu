#include <iostream>
#include <cublas_v2.h>
#include "Tensor.h"
#include "Node.h"
#include "time.h"


/*
==============================================================================
This function uses code from Simon Boehm's repository, "SGEMM_CUDA":
    https://github.com/siboehm/SGEMM_CUDA/tree/master
==============================================================================
*/

size_t Time::matmuls = 0;
float Tensor::memcpy_C_total_time = 0;
float Tensor::gemm_total_time = 0;
float Tensor::A_malloc_total_time = 0;
float Tensor::B_malloc_total_time = 0;
float Tensor::C_malloc_total_time = 0;
float Tensor::A_memcpy_total_time = 0;
float Tensor::B_memcpy_total_time = 0;
float Tensor::C_memset_total_time = 0;
float Tensor::sync_total_time = 0;
float Tensor::free_total_time = 0;
float Tensor::result_total_time = 0;

Tensor Tensor::matmul(Tensor& other, bool transpose_a, bool transpose_b, bool create_node) {
    Time::matmuls++;

    cudaSetDevice(0);

    static cublasHandle_t handle;
    static bool initialized = false;
    if (!initialized) {
        cublasCreate(&handle);
        initialized = true;
    }

    cublasOperation_t transa;
    cublasOperation_t transb;
    if (transpose_a) {
        transa = CUBLAS_OP_T;
    }
    else {
        transa = CUBLAS_OP_N;
    }
    if (transpose_b) {
        transb = CUBLAS_OP_T;
    }
    else {
        transb = CUBLAS_OP_N;
    }

    if (dimensions.size() == 1) {
        dimensions = {1, dimensions[0]};
        strides = {dimensions[0], 1};
    }
    if (other.dimensions.size() == 1) {
        other.dimensions = {1, other.dimensions[0]};
        other.strides = {other.dimensions[0], 1};
    }

    size_t A_rows_orig = dimensions[dimensions.size() - 2];
    size_t A_cols_orig = dimensions[dimensions.size() - 1];
    size_t B_rows_orig = other.dimensions[other.dimensions.size() - 2];
    size_t B_cols_orig = other.dimensions[other.dimensions.size() - 1];

    size_t effective_A_rows;
    size_t effective_A_cols;
    size_t effective_B_cols;
    if (transpose_a) {
        effective_A_rows = A_cols_orig;
        effective_A_cols = A_rows_orig;
    }
    else {
        effective_A_rows = A_rows_orig;
        effective_A_cols = A_cols_orig;
    }
    if (transpose_b) {
        effective_B_cols = B_rows_orig;
    }
    else {
        effective_B_cols = B_cols_orig;
    }

    size_t m_result, n_result, k_result;
    m_result = effective_A_rows;
    k_result = effective_A_cols;
    n_result = effective_B_cols;
    size_t m_cublas = n_result;
    size_t n_cublas = m_result;
    size_t k_cublas = k_result;

    size_t lda = B_cols_orig;
    size_t ldb = A_cols_orig;
    size_t ldc = n_result;

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

    size_t strideA = (A_rows_orig * A_cols_orig);
    size_t strideB = (B_rows_orig * B_cols_orig);
    size_t strideC = (m_cublas * n_cublas);
    if (A_batch_count == 1 and B_batch_count > 1) {
        strideA = 0;
    }
    else if (B_batch_count == 1 and A_batch_count > 1) {
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
        result_batch_dims = A_batch_dims;
    }

    // float *dA, *dB, *dC;
    // float *dC;

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
    float* C = new float[batch_count * m_result * n_result];

    size_t A_mem_size = sizeof(float) * A_rows_orig * A_cols_orig;
    size_t B_mem_size = sizeof(float) * B_rows_orig * B_cols_orig;
    size_t C_mem_size = sizeof(float) * batch_count * m_cublas * n_cublas;

    if (strideA > 0) {
        A_mem_size *= batch_count;
    }
    if (strideB > 0) {
        B_mem_size *= batch_count;
    }

    // ----------------------------------------------------------------------------------------------- 
    auto A_malloc_start = chrono::steady_clock::now();
    // cudaMalloc((void **)&dA, A_mem_size);
    auto A_malloc_stop = chrono::steady_clock::now();
    auto A_malloc_duration = chrono::duration_cast<chrono::microseconds>(A_malloc_stop - A_malloc_start);
    A_malloc_total_time += A_malloc_duration.count();
    if (Time::global_step == 2810) {
        cout << "Average A_malloc duration: " << A_malloc_total_time / Time::matmuls << " μs" << endl;
    }

    // ----------------------------------------------------------------------------------------------- 

    // ----------------------------------------------------------------------------------------------- 
    auto B_malloc_start = chrono::steady_clock::now();
    // cudaMalloc((void **)&dB, B_mem_size);
    auto B_malloc_stop = chrono::steady_clock::now();
    auto B_malloc_duration = chrono::duration_cast<chrono::microseconds>(B_malloc_stop - B_malloc_start);
    B_malloc_total_time += B_malloc_duration.count();
    if (Time::global_step == 2810) {
        cout << "Average B_malloc duration: " << B_malloc_total_time / Time::matmuls << " μs" << endl;
    }
    // ----------------------------------------------------------------------------------------------- 
    auto C_malloc_start = chrono::steady_clock::now();
    cudaMalloc((void **)&C, C_mem_size);
    auto C_malloc_stop = chrono::steady_clock::now();
    auto C_malloc_duration = chrono::duration_cast<chrono::microseconds>(C_malloc_stop - C_malloc_start);
    C_malloc_total_time += C_malloc_duration.count();
    if (Time::global_step == 2810) {
        cout << "Average C_malloc duration: " << C_malloc_total_time / Time::matmuls << " μs" << endl;
    }
    // ----------------------------------------------------------------------------------------------- 

    // ----------------------------------------------------------------------------------------------- 
    auto A_memcpy_start = chrono::steady_clock::now();
    // cudaMemcpy(dA, A, A_mem_size, cudaMemcpyHostToDevice);
    auto A_memcpy_stop = chrono::steady_clock::now();
    auto A_memcpy_duration = chrono::duration_cast<chrono::microseconds>(A_memcpy_stop - A_memcpy_start);
    A_memcpy_total_time += A_memcpy_duration.count();
    if (Time::global_step == 2810) {
        cout << "Average A_memcpy duration: " << A_memcpy_total_time / Time::matmuls << " μs" << endl;
    }
    // ----------------------------------------------------------------------------------------------- 

    // ----------------------------------------------------------------------------------------------- 
    auto B_memcpy_start = chrono::steady_clock::now();
    // cudaMemcpy(dB, B, B_mem_size, cudaMemcpyHostToDevice);
    auto B_memcpy_stop = chrono::steady_clock::now();
    auto B_memcpy_duration = chrono::duration_cast<chrono::microseconds>(B_memcpy_stop - B_memcpy_start);
    B_memcpy_total_time += B_memcpy_duration.count();
    if (Time::global_step == 2810) {
        cout << "Average B_memcpy duration: " << B_memcpy_total_time / Time::matmuls << " μs" << endl;
    }
    // ----------------------------------------------------------------------------------------------- 

    // ----------------------------------------------------------------------------------------------- 
    auto C_memset_start = chrono::steady_clock::now();
    cudaMemset(C, 0, C_mem_size);
    auto C_memset_stop = chrono::steady_clock::now();
    auto C_memset_duration = chrono::duration_cast<chrono::microseconds>(C_memset_stop - C_memset_start);
    C_memset_total_time += C_memset_duration.count();
    if (Time::global_step == 2810) {
        cout << "Average C_memset duration: " << C_memset_total_time / Time::matmuls << " μs" << endl;
    }
    // ----------------------------------------------------------------------------------------------- 

    float alpha = 1, beta = 0; // GEMM input parameters, C=α*AB+β*C

    // ----------------------------------------------------------------------------------------------- 
    auto gemm_start = chrono::steady_clock::now();
    cublasGemmStridedBatchedEx(handle,
                               transb,
                               transa,
                               m_cublas,
                               n_cublas,
                               k_cublas,
                               &alpha,
                               B,
                               CUDA_R_32F,
                               lda,
                               strideB,
                               A,
                               CUDA_R_32F,
                               ldb,
                               strideA,
                               &beta,
                               C,
                               CUDA_R_32F,
                               ldc,
                               strideC,
                               batch_count,
                               CUBLAS_COMPUTE_32F,
                               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    auto gemm_stop = chrono::steady_clock::now();
    auto gemm_duration = chrono::duration_cast<chrono::microseconds>(gemm_stop - gemm_start);
    gemm_total_time += gemm_duration.count();
    if (Time::global_step == 2810) {
        cout << "Average gemm duration: " << gemm_total_time / Time::matmuls << " μs" << endl;
    }
    // ----------------------------------------------------------------------------------------------- 
    
    // ----------------------------------------------------------------------------------------------- 
    auto memcpy_C_start = chrono::steady_clock::now();
    // cudaMemcpy(C, dC, C_mem_size, cudaMemcpyDeviceToHost);
    auto memcpy_C_stop = chrono::steady_clock::now();
    auto memcpy_C_duration = chrono::duration_cast<chrono::microseconds>(memcpy_C_stop - memcpy_C_start);
    memcpy_C_total_time += memcpy_C_duration.count();
    if (Time::global_step == 2810) {
        cout << "Average memcpy_C duration: " << memcpy_C_total_time / Time::matmuls << " μs" << endl;
    }
    // -----------------------------------------------------------------------------------------------

    // -----------------------------------------------------------------------------------------------
    auto free_start = chrono::steady_clock::now();
    // cudaFree(dA);
    // cudaFree(dB);
    // cudaFree(dC);
    auto free_stop = chrono::steady_clock::now();
    auto free_duration = chrono::duration_cast<chrono::microseconds>(free_stop - free_start);
    free_total_time += free_duration.count();
    if (Time::global_step == 2810) {
        cout << "Average free duration: " << free_total_time / Time::matmuls << " μs" << endl;
    }
    // -----------------------------------------------------------------------------------------------

    // -----------------------------------------------------------------------------------------------
    auto result_start = chrono::steady_clock::now();

    Tensor result;
    vector<size_t> result_dims;
    if (batch_count > 1) {
        result_dims.resize(result_batch_dims.size() + 2);
        copy(result_batch_dims.begin(), result_batch_dims.end(), result_dims.begin());
        result_dims[result_batch_dims.size()] = m_result;
        result_dims[result_batch_dims.size() + 1] = n_result;
    }
    else {
        result_dims = {m_result, n_result};
    }
    result.dimensions = result_dims;
    result.strides = compute_strides(result.dimensions);

    result.total_elements = batch_count * m_result * n_result;
    result.grad = shared_ptr<float>(new float[result.total_elements], default_delete<float[]>());
    fill(result.grad.get(), result.grad.get() + result.total_elements, 0);

    auto result_stop = chrono::steady_clock::now();
    auto result_duration = chrono::duration_cast<chrono::microseconds>(result_stop - result_start);
    result_total_time += result_duration.count();
    if (Time::global_step == 2810) {
        cout << "Average result duration: " << result_total_time / Time::matmuls << " μs" << endl;
    }
    // -----------------------------------------------------------------------------------------------

    // -----------------------------------------------------------------------------------------------

    // result.data = shared_ptr<float>(C, default_delete<float[]>());
    result.data = shared_ptr<float>(C, [](float* p) { cudaFree(p); });
    
    // -----------------------------------------------------------------------------------------------

    if (create_node) {
        if (requires_grad || other.requires_grad) {
            result.node = make_shared<MatmulBackward>(make_shared<Tensor>(*this), make_shared<Tensor>(other));
            result.node->tensor = &result;
        }
    }

    return result;
}
