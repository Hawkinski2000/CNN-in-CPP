#pragma once
#include <vector>
#include <memory>
#include <optional>
using namespace std;


class Node;

class Tensor {
    public:
        class TensorSlice {
            public:
                // Constructor for TensorSlice class used for chaining multiple [] operators
                TensorSlice(shared_ptr<float> data, const vector<size_t>& dimensions,
                    const vector<size_t>& strides, size_t offset, size_t level);

                // ---------------------------------------------------------------------------

                // Overload the [] operator for indexing into the tensor data
                TensorSlice operator[](size_t index);

                // Overload the = operator for assigning values to indices in the tensor
                TensorSlice& operator=(float value);

                // Overload the = operator for assigning values from another TensorSlice to indices in the tensor
                TensorSlice& operator=(const TensorSlice& other);

                // Overload the float reference conversion operator to return data after the final []
                operator float&();
        
            private:
                shared_ptr<float> data;
                vector<size_t> dimensions;
                vector<size_t> strides;
                size_t offset;
                size_t level;
        };

        // Default constructor for the Tensor class
        Tensor();

        // Copy constructor for the Tensor class
        Tensor(const Tensor& other);

        // Move constructor for the Tensor class
        Tensor(Tensor&& other);

        // Destructor for the Tensor class
        // ~Tensor();

        // ---------------------------------------------------------------------------

        // Function to create an empty tensor from a shape specified as an initializer_list
        static Tensor empty(initializer_list<size_t> dims, bool use_cuda=false);

        // Function to create an empty tensor from a shape specified as a vector
        static Tensor empty(vector<size_t> dims, bool use_cuda=false);

        // Function to create a tensor of zeros from a specified shape
        static Tensor zeros(initializer_list<size_t> dims, bool use_cuda=false);

        // Function to create tensor of zeros from a shape specified as a vector
        static Tensor zeros(vector<size_t> dims, bool use_cuda=false);

        // Function to create a tensor of ones from a specified shape
        static Tensor ones(initializer_list<size_t> dims, bool use_cuda=false);

        // Function to create tensor of ones from a shape specified as a vector
        static Tensor ones(vector<size_t> dims);

        // Function to create a tensor of random values from a specified shape
        static Tensor rand(initializer_list<size_t> dims, size_t in_features = 0, bool use_cuda=false);

        // Function to create a tensor of random values from a specified shape that runs on the GPU
        static Tensor cuda_rand(initializer_list<size_t> dims, size_t in_features = 0);

        // Function to create tensor of random values from a shape specified as a vector
        static Tensor rand(vector<size_t> dims, size_t in_features = 0);

        // Function to create a tensor from specified values
        static Tensor tensor(initializer_list<float> values, bool use_cuda=false);

        // Function to create a tensor from a vector
        static Tensor tensor(vector<float>& values, bool use_cuda=false);

        // ---------------------------------------------------------------------------

        // Overload the [] operator for indexing into the tensor data
        TensorSlice operator[](size_t index);

        // Overload the = operator to move a temporary tensor into an existing tensor
        Tensor& operator=(Tensor&& other);

        // Overload the = operator for copying one tensor to another
        Tensor& operator=(const Tensor& other);

        // ---------------------------------------------------------------------------

        // Function to return a new tensor with the same data as the original tensor but of a different shape
        Tensor view(initializer_list<int> shape);

        // Function to flatten a tensor by reshaping it into a one-dimensional tensor
        Tensor flatten();

        // Function to return a tensor that is a transposed version of a tensor
        Tensor transpose(size_t dim0, size_t dim1);

        // ---------------------------------------------------------------------------

        // Function to return the sum of all elements in a tensor
        Tensor sum(optional<size_t> dim = nullopt);

        // Function to return the sum of all elements in a tensor that runs on the GPU
        Tensor cuda_sum(optional<size_t> dim = nullopt);

        // Function to return the mean of all elements in a tensor
        Tensor mean();

        // Function for element-wise powers between tensors and scalars
        Tensor pow(int exponent);

        // Function to return the exponential of all elements in a tensor
        Tensor exp();

        // Function to return the exponential of all elements in a tensor that runs on the GPU
        Tensor cuda_exp();

        // Function to return the natural logarithm of all elements in a tensor
        Tensor log();

        // Function to return the natural logarithm of all elements in a tensor that runs on the GPU
        Tensor cuda_log();

        // Function to return the minimum value of all elements in a tensor
        Tensor min(optional<size_t> dim = nullopt);

        // Function to return the maximum value of all elements in a tensor
        Tensor max(optional<size_t> dim = nullopt);

        // Function to return the maximum value of all elements in a tensor that runs on the GPU
        Tensor cuda_max(optional<size_t> dim = nullopt);

        // Function to return the indices of the maximum value of all elements in a tensor
        Tensor argmax(optional<size_t> dim = nullopt);

        // Function to return the indices of the maximum value of all elements in a tensor that runs on the GPU
        Tensor cuda_argmax(optional<size_t> dim = nullopt);

        // Function to return true if two tensors have the same shape and elements, otherwise false.
        bool equal(const Tensor& other);

        // Overload the == operator for element-wise equality between tensors
        Tensor operator==(const Tensor& other);
        
        // Function for element-wise equality between tensors that runs on the GPU
        Tensor cuda_eq(const Tensor& other);

        // Function to return the matrix product of two tensors.
        // This function uses code from Simon Boehm's repository, "SGEMM_CUDA":
        // https://github.com/siboehm/SGEMM_CUDA/tree/master
        Tensor matmul(Tensor& other, bool transpose_a=false, bool transpose_b=false, bool create_node=true);
        static float memcpy_C_total_time;
        static float gemm_total_time;
        static float A_malloc_total_time;
        static float B_malloc_total_time;
        static float C_malloc_total_time;
        static float A_memcpy_total_time;
        static float B_memcpy_total_time;
        static float C_memset_total_time;
        static float sync_total_time;
        static float free_total_time;
        static float result_total_time;

        // ---------------------------------------------------------------------------

        // Function to compute a tensor's strides from its dimensions
        static vector<size_t> compute_strides(const vector<size_t>& dimensions);

        // Function to pad a shape or strides vector with 1's or 0's for broadcasting
        vector<size_t> pad_vector(const vector<size_t>& original, size_t num_dims, size_t pad_with);

        // Function to return a vector containing the resulting tensor shape from an operation on two broadcastable tensors
        vector<size_t> broadcast_result_shape(const vector<size_t>& a_dims, const vector<size_t>& b_dims);

        // Function to return a vector containing the broadcastable strides of a tensor given a resulting tensor shape
        vector<size_t> broadcast_strides(const vector<size_t>& original_dims, const vector<size_t>& original_strides, const vector<size_t>& result_dims);

        // Function to compute the memory offset for a multi-dimensional index from the tensor's strides
        size_t compute_offset(const vector<size_t>& indices, const vector<size_t>& strides);

        // Function to advance a multi-dimensional index in a tensor. Returns true if the index was successfully incremented, or false if iteration is complete
        bool next_index(vector<size_t>& indices, const vector<size_t>& result_dims);

        // ---------------------------------------------------------------------------

        // Overload the + operator for element-wise addition between tensors
        Tensor operator+(Tensor& other);

        // Function for element-wise addition between tensors that runs on the GPU
        Tensor cuda_add(Tensor& other);

        // Overload the + operator for element-wise addition between tensors and scalars
        Tensor operator+(float value);

        // Function for element-wise addition between tensors and scalars that runs on the GPU
        Tensor cuda_add_scalar(float value);

        // Overload the += operator for element-wise addition and assignment between tensors
        Tensor& operator+=(const Tensor& other);

        // Function for element-wise addition and assignment between tensors that runs on the GPU
        Tensor& cuda_add_(const Tensor& other);

        // Overload the += operator for element-wise addition and assignment between tensors and scalars
        Tensor& operator+=(float value);

        // Function for element-wise addition and assignment between tensors and scalars that runs on the GPU
        Tensor& cuda_add_scalar_(float value);

        // ---------------------------------------------------------------------------

        // Overload the - operator for element-wise subtraction between tensors
        Tensor operator-(Tensor& other);

        // Function for element-wise subtraction between tensors that runs on the GPU
        Tensor cuda_sub(Tensor& other);

        // Overload the - operator for element-wise subtraction between tensors and scalars
        Tensor operator-(float value);

        // Function for element-wise subtraction between tensors and scalars that runs on the GPU
        Tensor cuda_sub_scalar(float value);

        // Overload the -= operator for element-wise subtraction and assignment between tensors
        Tensor& operator-=(const Tensor& other);

        // Function for element-wise subtraction and assignment between tensors that runs on the GPU
        Tensor& cuda_sub_(const Tensor& other);

        // Overload the -= operator for element-wise subtraction and assignment between tensors and scalars
        Tensor& operator-=(float value);

        // Function for element-wise subtraction and assignment between tensors and scalars that runs on the GPU
        Tensor& cuda_sub_scalar_(float value);

        // ---------------------------------------------------------------------------

        // Overload the * operator for element-wise multiplication between tensors
        Tensor operator*(Tensor& other);

        // Function for element-wise multiplication between tensors that runs on the GPU
        Tensor cuda_mul(Tensor& other);

        // Overload the * operator for element-wise multiplication between tensors and scalars
        Tensor operator*(float value);

        // Function for element-wise multiplication between tensors and scalars that runs on the GPU
        Tensor cuda_mul_scalar(float value);

        // Overload the *= operator for element-wise multiplication and assignment between tensors
        Tensor& operator*=(const Tensor& other);

        // Function for element-wise multiplication and assignment between tensors that runs on the GPU
        Tensor& cuda_mul_(const Tensor& other);

        // Overload the *= operator for element-wise multiplication and assignment between tensors and scalars
        Tensor& operator*=(float value);

        // Function for element-wise multiplication and assignment between tensors and scalars that runs on the GPU
        Tensor& cuda_mul_scalar_(float value);

        // ---------------------------------------------------------------------------

        // Overload the / operator for element-wise division between tensors
        Tensor operator/(Tensor& other);

        // Function for element-wise division between tensors that runs on the GPU
        Tensor cuda_div(Tensor& other);

        // Overload the / operator for element-wise division between tensors and scalars
        Tensor operator/(float value);

        // Function for element-wise division between tensors and scalars that runs on the GPU
        Tensor cuda_div_scalar(float value);

        // Overload the /= operator for element-wise division and assignment between tensors
        Tensor& operator/=(const Tensor& other);

        // Function for element-wise division and assignment between tensors that runs on the GPU
        Tensor& cuda_div_(const Tensor& other);

        // Overload the /= operator for element-wise division and assignment between tensors and scalars
        Tensor& operator/=(float value);

        // Function for element-wise division and assignment between tensors and scalars that runs on the GPU
        Tensor& cuda_div_scalar_(float value);

        // ---------------------------------------------------------------------------

        // Overload the << operator for printing the contents of a tensor
        friend ostream& operator<<(ostream& os, const Tensor& tensor);

        // Function to return a tensor's shape vector
        const vector<size_t>& shape() const;

        // Function to return a tensor's shape as a string
        string shape_str();

        // Function to return the total number of elements in a tensor
        size_t numel();

        // Function to return the elements in a tensor
        const float* get_data() const;

        // Function to run tests on the Tensor class
        static void run_tests();

        // ---------------------------------------------------------------------------

        // Function to call Engine::run_backward() to compute the gradient of the current tensor w.r.t. graph leaves.
        void backward();
        
        // ---------------------------------------------------------------------------

        // Function to apply the rectified linear unit function to the input tensor
        friend Tensor relu(const Tensor& input);

        // Function to apply the rectified linear unit function to the input tensor that runs on the GPU.
        friend Tensor relu_cuda(const Tensor& input);

        // Function to apply softmax to the input tensor
        friend Tensor softmax(Tensor& input, optional<size_t> dim);

        // Function to apply log softmax to the input tensor
        friend Tensor log_softmax(Tensor& input, optional<size_t> dim);

        // Function to compute the negative log likelihood loss from the input tensor and targets
        friend Tensor nll_loss(Tensor& input, Tensor& targets);

        // Function to compute the negative log likelihood loss from the input tensor and targets that runs on the GPU.
        friend Tensor nll_loss_cuda(Tensor& input, Tensor& targets);

        // Function to compute the cross entropy loss between input tensor and targets
        friend Tensor cross_entropy(Tensor& input, Tensor& targets);

        // Function to extract sliding local blocks from a batched input tensor
        friend Tensor unfold(Tensor& input, initializer_list<size_t> kernel_size, size_t dilation, size_t padding, size_t stride);

        // Function to extract sliding local blocks from a batched input tensor that runs on the GPU.
        friend Tensor unfold_cuda(Tensor input, initializer_list<size_t> kernel_size, size_t dilation, size_t padding, size_t stride);

        // Function to combine an array of sliding local blocks into a large containing tensor
        friend Tensor fold(Tensor& input, initializer_list<size_t> output_size, initializer_list<size_t> kernel_size, size_t dilation, size_t padding, size_t stride);

        // Function to combine an array of sliding local blocks into a large containing tensor that runs on the GPU.
        friend Tensor fold_cuda(Tensor& input, initializer_list<size_t> output_size, initializer_list<size_t> kernel_size, size_t dilation, size_t padding, size_t stride);

        // Function to apply a 2D max pooling over an input tensor
        friend Tensor maxpool2d_cuda(Tensor input, initializer_list<size_t> kernel_size, size_t stride, size_t padding, size_t dilation);

        // ---------------------------------------------------------------------------

        shared_ptr<float> data;
        string device = "cpu";
        float* device_data = nullptr;
        bool requires_grad = true;
        shared_ptr<float> grad = nullptr;
        shared_ptr<Node> node = nullptr;
        
    private:
        friend class Node;
        friend class AddBackward;
        friend class SubBackward;
        friend class MulBackward;
        friend class DivBackward;
        friend class MatmulBackward;
        friend class ReLUBackward;
        friend class LogSoftmaxBackward;
        friend class NLLLossBackward;
        friend class Conv2dBackward;
        friend class MaxPool2dBackward;

        // ---------------------------------------------------------------------------

        // Constructor for Tensor class used by creation methods that specify a shape as an initializer_list
        Tensor(initializer_list<size_t> dims, bool use_cuda=false);

        // Constructor for Tensor class used by creation methods that specify a shape as a vector
        Tensor(const vector<size_t>& dims, bool use_cuda=false);

        // Constructor for Tensor class used by tensor() where values are specified
        Tensor(initializer_list<float> values, bool use_cuda=false);

        // Constructor for Tensor class used by tensor() to convert a vector to a tensor
        Tensor(const vector<float>& values, bool use_cuda=false);

        // ---------------------------------------------------------------------------

        // shared_ptr<float> data;
        vector<size_t> dimensions;
        vector<size_t> strides;
        size_t total_elements;
};
