#include <iostream>
#include <cmath>
#include "Node.h"
#include "Tensor.h"
using namespace std;


// Destructor for the Node class
Node::~Node() = default;

// ---------------------------------------------------------------------------

// Constructor for the AddBackward class
AddBackward::AddBackward(shared_ptr<Tensor> a, shared_ptr<Tensor> b) : lhs(a), rhs(b) {
    if (lhs->node) {
        children.push_back({lhs->node});
    }
    if (rhs->node) {
        children.push_back({rhs->node});
    }
}

// Function to propagate gradients backward to child nodes
void AddBackward::backward() {
    for (size_t i = 0; i < tensor->total_elements; i++) {
        lhs->grad.get()[i % lhs->total_elements] += tensor->grad.get()[i];
        rhs->grad.get()[i % rhs->total_elements] += tensor->grad.get()[i];
    }
}

// Function to return the type of Node
string AddBackward::name() {
    return "AddBackward";
}

// ---------------------------------------------------------------------------

// Constructor for the SubBackward class
SubBackward::SubBackward(shared_ptr<Tensor> a, shared_ptr<Tensor> b) : lhs(a), rhs(b) {
    if (lhs->node) {
        children.push_back({lhs->node});
    }
    if (rhs->node) {
        children.push_back({rhs->node});
    }
}

// Function to propagate gradients backward to child nodes
void SubBackward::backward() {
    for (size_t i = 0; i < tensor->total_elements; i++) {
        lhs->grad.get()[i] += tensor->grad.get()[i];
        rhs->grad.get()[i] -= tensor->grad.get()[i];
    }
}

// Function to return the type of Node
string SubBackward::name() {
    return "SubBackward";
}

// ---------------------------------------------------------------------------

// Constructor for the MulBackward class
MulBackward::MulBackward(shared_ptr<Tensor> a, shared_ptr<Tensor> b) : lhs(a), rhs(b) {
    if (lhs->node) {
        children.push_back({lhs->node});
    }
    if (rhs->node) {
        children.push_back({rhs->node});
    }
}

// Function to propagate gradients backward to child nodes
void MulBackward::backward() {
    for (size_t i = 0; i < tensor->total_elements; i++) {
        lhs->grad.get()[i] += tensor->grad.get()[i] * rhs->get_data()[i];
        rhs->grad.get()[i] += tensor->grad.get()[i] * lhs->get_data()[i];
    }
}

// Function to return the type of Node
string MulBackward::name() {
    return "MulBackward";
}

// ---------------------------------------------------------------------------

// Constructor for the DivBackward class
DivBackward::DivBackward(shared_ptr<Tensor> a, shared_ptr<Tensor> b) : lhs(a), rhs(b) {
    if (lhs->node) {
        children.push_back({lhs->node});
    }
    if (rhs->node) {
        children.push_back({rhs->node});
    }
}

// Function to propagate gradients backward to child nodes
void DivBackward::backward() {
    for (size_t i = 0; i < tensor->total_elements; i++) {
        lhs->grad.get()[i] += tensor->grad.get()[i] * (1 / rhs->get_data()[i]);
        rhs->grad.get()[i] += tensor->grad.get()[i] * (-lhs->get_data()[i] / pow(rhs->get_data()[i], 2));
    }
}

// Function to return the type of Node
string DivBackward::name() {
    return "DivBackward";
}

// ---------------------------------------------------------------------------

// Constructor for the MatmulBackward class
MatmulBackward::MatmulBackward(shared_ptr<Tensor> a, shared_ptr<Tensor> b) : lhs(a), rhs(b) {
    if (lhs->node) {
        children.push_back({lhs->node});
    }
    if (rhs->node) {
        children.push_back({rhs->node});
    }
}

// Function to propagate gradients backward to child nodes
void MatmulBackward::backward() {
    Tensor dLdc = *tensor;
    copy(tensor->grad.get(), tensor->grad.get() + tensor->total_elements, dLdc.data.get());

    Tensor dLda = dLdc.matmul(*rhs, false, true, false);
    Tensor dLdb = lhs->matmul(dLdc, true, false, false);

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

    // A and B were both broadcast
    if (C_batch_count > A_batch_count && C_batch_count > B_batch_count) {
        if (lhs->requires_grad) {
            for (size_t i = 0; i < C_batch_count * m * k; i++) {
                lhs->grad.get()[i % lhs->total_elements] += dLda.data.get()[i % lhs->total_elements];
            }
        }
        if (rhs->requires_grad) {
            for (size_t i = 0; i < C_batch_count * k * n; i++) {
                rhs->grad.get()[i % rhs->total_elements] += dLdb.data.get()[i % rhs->total_elements];
            }
        }
    }

    // Only B was broadcast
    else if (A_batch_count > B_batch_count) {
        if (lhs->requires_grad) {
            for (size_t i = 0; i < lhs->total_elements; i++) {
                lhs->grad.get()[i] += dLda.data.get()[i];
            }
        }
        if (rhs->requires_grad) {
            for (size_t i = 0; i < C_batch_count * k * n; i++) {
                rhs->grad.get()[i % rhs->total_elements] += dLdb.data.get()[i % rhs->total_elements];
            }
        }
    }

    // Only A was broadcast
    else if (B_batch_count > A_batch_count) {
        if (lhs->requires_grad) {
            for (size_t i = 0; i < C_batch_count * m * k; i++) {
                lhs->grad.get()[i % lhs->total_elements] += dLda.data.get()[i % lhs->total_elements];
            }
        }
        if (rhs->requires_grad) {
            for (size_t i = 0; i < rhs->total_elements; i++) {
                rhs->grad.get()[i] += dLdb.data.get()[i];
            }
        }
    }

    // No broadcasting
    else {
        if (lhs->requires_grad) {
            for (size_t i = 0; i < lhs->total_elements; i++) {
                lhs->grad.get()[i] += dLda.data.get()[i];
            }
        }
        if (rhs->requires_grad) {
            for (size_t i = 0; i < rhs->total_elements; i++) {
                rhs->grad.get()[i] += dLdb.data.get()[i];
            }
        }
    }
}

// Function to return the type of Node
string MatmulBackward::name() {
    return "MatmulBackward";
}

// ---------------------------------------------------------------------------

// Constructor for the ReLUBackward class
ReLUBackward::ReLUBackward(shared_ptr<Tensor> input) : input(input) {
    if (input->node) {
        children.push_back({input->node});
    }
}

// Function to propagate gradients backward to child nodes
void ReLUBackward::backward() {
    Tensor dLdy = *tensor;
    copy(tensor->grad.get(), tensor->grad.get() + tensor->total_elements, dLdy.data.get());

    for (size_t i = 0; i < input->total_elements; i++) {
        if (input->data.get()[i] > 0) {
            input->grad.get()[i] = dLdy.grad.get()[i];
        }
        else {
            input->grad.get()[i] = 0;
        }
    }
}

// Function to return the type of Node
string ReLUBackward::name() {
    return "ReLUBackward";
}

// ---------------------------------------------------------------------------

// Constructor for the LogSoftmaxBackward class
LogSoftmaxBackward::LogSoftmaxBackward(shared_ptr<Tensor> input, shared_ptr<Tensor> softmax_values) : input(input), softmax_values(softmax_values) {
    if (input->node) {
        children.push_back({input->node});
    }
}

// Function to propagate gradients backward to child nodes
void LogSoftmaxBackward::backward() {
    Tensor dLdy = *tensor;
    copy(tensor->grad.get(), tensor->grad.get() + tensor->total_elements, dLdy.data.get());
    Tensor sum_dLdy = dLdy.sum(1);
    Tensor sum_dLdy_p = sum_dLdy * *softmax_values;
    dLdy.requires_grad = false;
    Tensor dLdx = dLdy - sum_dLdy_p;
    for (size_t i = 0; i < input->total_elements; i++) {
        input->grad.get()[i] += dLdx.data.get()[i];
    }
}

// Function to return the type of Node
string LogSoftmaxBackward::name() {
    return "LogSoftmaxBackward";
}

// ---------------------------------------------------------------------------

// Constructor for the NLLLossBackward class
NLLLossBackward::NLLLossBackward(shared_ptr<Tensor> input, shared_ptr<Tensor> targets) : input(input), targets(targets) {
    if (input->node) {
        children.push_back({input->node});
    }
}

// Function to propagate gradients backward to child nodes
void NLLLossBackward::backward() {
    size_t batch_size = input->dimensions[0];
    size_t num_classes = input->dimensions[1];
    
    for (size_t i = 0; i < batch_size; i++) {
        size_t target_class = static_cast<size_t>(targets->data.get()[i]);
        size_t idx = i * num_classes + target_class;
        input->grad.get()[idx] = -1 / static_cast<float>(batch_size);
    }
}

// Function to return the type of Node
string NLLLossBackward::name() {
    return "NLLLossBackward";
}

// ---------------------------------------------------------------------------

// Constructor for the Conv2dBackward class
Conv2dBackward::Conv2dBackward(shared_ptr<Tensor> input,
                               shared_ptr<Tensor> weight,
                               shared_ptr<Tensor> inp_unf,
                               initializer_list<size_t> kernel_size,
                               size_t stride,
                               size_t padding,
                               size_t dilation)
    : input(input),
      weight(weight),
      inp_unf(inp_unf),
      kernel_size(kernel_size),
      stride(stride),
      padding(padding),
      dilation(dilation) {
    if (input->node) {
        children.push_back({input->node});
    }
    if (weight->node) {
        children.push_back({weight->node});
    }
}

// Function to propagate gradients backward to child nodes
void Conv2dBackward::backward() {
    Tensor dLdc = *tensor;
    copy(tensor->grad.get(), tensor->grad.get() + tensor->total_elements, dLdc.data.get());

    int N = input->dimensions[0]; // Batch size

    int out_channels = weight->dimensions[0];

    dLdc = dLdc.view({N, out_channels, -1});
    
    Tensor dLdw = inp_unf->matmul(dLdc, false, true, false);
    Tensor w = weight->view({out_channels, -1});
    Tensor dLdx_unf = dLdc.matmul(w, true, false, false);

    size_t kH = *kernel_size.begin();
    size_t kW;
    if (kernel_size.size() == 1) {
        kW = kH;
    }
    else {
        kW = *(kernel_size.begin() + 1);
    }
    dLdx_unf = dLdx_unf.transpose(1, 2);
    Tensor dLdx = fold(dLdx_unf, {input->dimensions[2], input->dimensions[3]}, {kH, kW}, dilation, padding, stride);

    for (size_t i = 0; i < dLdw.total_elements; i++) {
        weight->grad.get()[i % weight->total_elements] += dLdw.data.get()[i];
    }
    for (size_t i = 0; i < dLdx.total_elements; i++) {
        input->grad.get()[i % input->total_elements] += dLdx.data.get()[i];
    }
}

// Function to return the type of Node
string Conv2dBackward::name() {
    return "Conv2dBackward";
}
