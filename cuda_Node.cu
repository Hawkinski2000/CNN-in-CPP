#include <iostream>
#include "Node.h"
#include "Tensor.h"


// Kernel for cuda_backward() of AddBackward nodes
__global__ void add_backward_kernel(const float* __restrict__ grad_output,
                                    float* lhs_grad,
                                    float* rhs_grad,
                                    size_t lhs_size,
                                    size_t rhs_size,
                                    size_t total_elements) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x; // Index of element in grad_output to accumulate to
    if (i >= total_elements) {
        return;
    }

    float grad_val = grad_output[i];

    atomicAdd(&lhs_grad[i % lhs_size], grad_val);
    atomicAdd(&rhs_grad[i % rhs_size], grad_val);
}

// Function to propagate gradients backward to child nodes that runs on the GPU
void AddBackward::cuda_backward() {
    int threads = 256;
    int blocks = (tensor->total_elements + threads - 1) / threads;
    add_backward_kernel<<<blocks, threads>>>(tensor->grad.get(),
                                             lhs->grad.get(),
                                             rhs->grad.get(),
                                             lhs->total_elements,
                                             rhs->total_elements,
                                             tensor->total_elements);

    cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------------

// Kernel for cuda_backward() of SubBackward nodes
__global__ void sub_backward_kernel(const float* __restrict__ grad_output,                            
                                    float* lhs_grad,
                                    float* rhs_grad,
                                    size_t lhs_size,
                                    size_t rhs_size,
                                    size_t total_elements) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x; // Index of element in grad_output to accumulate to
    if (i >= total_elements) {
        return;
    }

    float grad_val = grad_output[i];
    
    atomicAdd(&lhs_grad[i % lhs_size], grad_val);
    atomicAdd(&rhs_grad[i % rhs_size], -grad_val);
}

// Function to propagate gradients backward to child nodes that runs on the GPU
void SubBackward::cuda_backward() {
    int threads = 256;
    int blocks = (tensor->total_elements + threads - 1) / threads;
    sub_backward_kernel<<<blocks, threads>>>(tensor->grad.get(),
                                             lhs->grad.get(),
                                             rhs->grad.get(),
                                             lhs->total_elements,
                                             rhs->total_elements,
                                             tensor->total_elements);

    cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------------

// Kernel for cuda_backward() of MulBackward nodes
__global__ void mul_backward_kernel(const float* __restrict__ grad_output,
                                    const float* __restrict__ lhs_data,
                                    const float* __restrict__ rhs_data,
                                    float* lhs_grad,
                                    float* rhs_grad,
                                    size_t lhs_size,
                                    size_t rhs_size,
                                    size_t total_elements) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x; // Index of element in grad_output to accumulate to
    if (i >= total_elements) {
        return;
    }

    float grad_val = grad_output[i];

    atomicAdd(&lhs_grad[i % lhs_size], grad_val * lhs_data[i % lhs_size]);
    atomicAdd(&rhs_grad[i % rhs_size], grad_val * rhs_data[i % rhs_size]);
}

// Function to propagate gradients backward to child nodes that runs on the GPU
void MulBackward::cuda_backward() {
    int threads = 256;
    int blocks = (tensor->total_elements + threads - 1) / threads;
    mul_backward_kernel<<<blocks, threads>>>(tensor->grad.get(),
                                             lhs->data.get(),
                                             rhs->data.get(),
                                             lhs->grad.get(),
                                             rhs->grad.get(),
                                             lhs->total_elements,
                                             rhs->total_elements,
                                             tensor->total_elements);

    cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------------

// Kernel for cuda_backward() of DivBackward nodes
__global__ void div_backward_kernel(const float* __restrict__ grad_output,
                                    float* lhs_grad,
                                    float* rhs_grad,
                                    size_t lhs_size,
                                    size_t rhs_size,
                                    size_t total_elements) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x; // Index of element in grad_output to accumulate to
    if (i >= total_elements) {
        return;
    }

    // TODO

    return;
}

// Function to propagate gradients backward to child nodes that runs on the GPU
void DivBackward::cuda_backward() {
    int threads = 256;
    int blocks = (tensor->total_elements + threads - 1) / threads;
    div_backward_kernel<<<blocks, threads>>>(tensor->grad.get(),
                                             lhs->grad.get(),
                                             rhs->grad.get(),
                                             lhs->total_elements,
                                             rhs->total_elements,
                                             tensor->total_elements);

    cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------------

// Kernel for cuda_backward() of MatmulBackward nodes
__global__ void matmul_backward_kernel(const float* __restrict__ dLda,
                                       const float* __restrict__ dLdb,
                                       float* lhs_grad,
                                       float* rhs_grad,
                                       size_t lhs_size,
                                       size_t rhs_size,
                                       size_t lhs_elements,
                                       size_t rhs_elements,
                                       bool lhs_broadcast,
                                       bool rhs_broadcast) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < lhs_elements) {
        if (lhs_broadcast) {
            atomicAdd(&lhs_grad[i % lhs_size], dLda[i]);
        }
        else if (i < lhs_size) {
            lhs_grad[i] += dLda[i];
        }
    }

    if (i < rhs_elements) {
        if (rhs_broadcast) {
            atomicAdd(&rhs_grad[i % rhs_size], dLdb[i]);
        }
        else if (i < rhs_size) {
            rhs_grad[i] += dLdb[i];
        }
    }
}

// Function to propagate gradients backward to child nodes that runs on the GPU
void MatmulBackward::cuda_backward() {
    size_t A_batch_count = 1;
    if (lhs->dimensions.size() > 2) {
        for (size_t i = 0; i < lhs->dimensions.size() - 2; i++) {
            A_batch_count *= lhs->dimensions[i];
        }
    }
    size_t B_batch_count = 1;
    if (rhs->dimensions.size() > 2) {
        for (size_t i = 0; i < rhs->dimensions.size() - 2; i++) {
            B_batch_count *= rhs->dimensions[i];
        }
    }
    size_t C_batch_count = 1;
    if (tensor->dimensions.size() > 2) {
        for (size_t i = 0; i < tensor->dimensions.size() - 2; i++) {
            C_batch_count *= tensor->dimensions[i];
        }
    }

    size_t m = lhs->dimensions[lhs->dimensions.size() - 2];
    size_t k = lhs->dimensions[lhs->dimensions.size() - 1];
    size_t n = rhs->dimensions[rhs->dimensions.size() - 1];

    size_t lhs_elements = 0;
    size_t rhs_elements = 0;
    bool lhs_broadcast = false;
    bool rhs_broadcast = false;

    // A and B were both broadcast
    if (C_batch_count > A_batch_count && C_batch_count > B_batch_count) {
        lhs_broadcast = true;
        rhs_broadcast = true;
        lhs_elements = C_batch_count * m * k;
        rhs_elements = C_batch_count * k * n;
    }

    // Only B was broadcast
    else if (A_batch_count > B_batch_count) {
        lhs_broadcast = false;
        rhs_broadcast = true;
        lhs_elements = lhs->total_elements;
        rhs_elements = C_batch_count * k * n;
    }

    // Only A was broadcast
    else if (B_batch_count > A_batch_count) {
        lhs_broadcast = true;
        rhs_broadcast = false;
        lhs_elements = C_batch_count * m * k;
        rhs_elements = rhs->total_elements;
    }

    // No broadcasting
    else {
        lhs_broadcast = false;
        rhs_broadcast = false;
        lhs_elements = lhs->total_elements;
        rhs_elements = rhs->total_elements;
    }

    size_t max_elements = max(lhs_elements, rhs_elements);
    int threads = 256;
    int blocks = (max_elements + threads - 1) / threads;
    matmul_backward_kernel<<<blocks, threads>>>(dLda.data.get(),
                                                dLdb.data.get(),
                                                lhs->grad.get(),
                                                rhs->grad.get(),
                                                lhs->total_elements,
                                                rhs->total_elements,
                                                lhs_elements,
                                                rhs_elements,
                                                lhs_broadcast,
                                                rhs_broadcast);

    cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------------

// Kernel for cuda_backward() of ReLUBackward nodes
__global__ void relu_backward_kernel(const float* __restrict__ input_data,
                                     const float* __restrict__ grad_output,
                                     float* input_grad,
                                     size_t total_elements) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x; // Index of element in grad_output to accumulate to
    if (i >= total_elements) {
        return;
    }

    if (input_data[i] > 0.0f) {
        atomicAdd(&input_grad[i], grad_output[i]);
    }
}

// Function to propagate gradients backward to child nodes that runs on the GPU
void ReLUBackward::cuda_backward() {
    int threads = 256;
    int blocks = (input->total_elements + threads - 1) / threads;
    relu_backward_kernel<<<blocks, threads>>>(input->data.get(),
                                              tensor->grad.get(),
                                              input->grad.get(),
                                              input->total_elements);

    cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------------

// Kernel for cuda_backward() of LogSoftmaxBackward nodes
__global__ void log_softmax_backward_kernel(const float* __restrict__ dLdx, float* __restrict__ grad, size_t total_elements) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < total_elements) {
        atomicAdd(&grad[i], dLdx[i]);
    }
}

// Function to propagate gradients backward to child nodes that runs on the GPU
void LogSoftmaxBackward::cuda_backward() {
    Tensor dLdy = *tensor;
    dLdy.data = tensor->grad;

    Tensor sum_dLdy = dLdy.sum(1);

    sum_dLdy.requires_grad = false;
    Tensor sum_dLdy_p = sum_dLdy * *softmax_values;

    dLdy.requires_grad = false;
    Tensor dLdx = dLdy - sum_dLdy_p;

    int threads = 256;
    int blocks = (input->total_elements + threads - 1) / threads;
    log_softmax_backward_kernel<<<blocks, threads>>>(dLdx.data.get(), input->grad.get(), input->total_elements);

    cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------------

// Kernel for cuda_backward() of NLLLossBackward nodes
__global__ void nll_loss_backward_kernel(const float* __restrict__ targets,
                                         float* __restrict__ grad,
                                         size_t batch_size,
                                         size_t num_classes) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < batch_size) {
        size_t y_i = static_cast<size_t>(targets[i]);
        size_t idx = i * num_classes + y_i;
        atomicAdd(&grad[idx], -1.0f / static_cast<float>(batch_size));
    }
}

// Function to propagate gradients backward to child nodes that runs on the GPU
void NLLLossBackward::cuda_backward() {
    size_t batch_size = input->dimensions[0];
    size_t num_classes = input->dimensions[1];

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    nll_loss_backward_kernel<<<blocks, threads>>>(targets->data.get(),
                                                  input->grad.get(),
                                                  batch_size,
                                                  num_classes);

    cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------------

// Kernel for cuda_backward() of Conv2dBackward nodes
__global__ void conv2d_weight_backward_kernel(float* dLdw, float* w_grad, size_t w_size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < w_size) {
        atomicAdd(&w_grad[i], dLdw[i]);
    }
}

__global__ void conv2d_input_backward_kernel(float* dLdx, float* x_grad, size_t x_size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < x_size) {
        atomicAdd(&x_grad[i], dLdx[i]);
    }
}


// Function to propagate gradients backward to child nodes that runs on the GPU
void Conv2dBackward::cuda_backward() {
    int threads = 256;

    int weight_blocks = (dLdw.total_elements + threads - 1) / threads;
    conv2d_weight_backward_kernel<<<weight_blocks, threads>>>(dLdw.data.get(),
                                                              weight->grad.get(),
                                                              dLdw.total_elements);

    int input_blocks = (dLdx.total_elements + threads - 1) / threads;
    conv2d_input_backward_kernel<<<input_blocks, threads>>>(dLdx.data.get(),
                                                            input->grad.get(),
                                                            dLdx.total_elements);

    cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------------

// Kernel for cuda_backward() of MaxPool2dBackward nodes
__global__ void maxpool2d_backward_kernel(const float* __restrict__ grad_output,
                                          const float* __restrict__ max_indices,
                                          float* __restrict__ grad_input,
                                          size_t total_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_elements) {
        return;
    }

    int max_idx = static_cast<int>(__float2int_rn(max_indices[idx]));

    atomicAdd(&grad_input[max_idx], grad_output[idx]);
}

// Function to propagate gradients backward to child nodes that runs on the GPU
void MaxPool2dBackward::cuda_backward() {
    int threads = 256;
    int blocks = (tensor->total_elements + threads - 1) / threads;

    maxpool2d_backward_kernel<<<blocks, threads>>>(tensor->grad.get(),
                                                   max_indices->data.get(),
                                                   input->grad.get(),
                                                   tensor->total_elements);

    cudaDeviceSynchronize();
}
