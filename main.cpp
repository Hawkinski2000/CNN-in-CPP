#include <iostream>
#include <fstream>
#include <iomanip>
#include <cuda_runtime.h>
#include <chrono>
#include "Tensor.h"
#include "Engine.h"
#include "nn.h"
#include "functional.h"
#include "optim.h"
#include "time.h"
using namespace std;


// Compile with: nvcc -o main main.cpp Tensor.cpp Node.cpp Engine.cpp nn.cpp functional.cpp optim.cpp matmul.cu unfold.cu fold.cu maxpool2d.cu cuda_utils.cu add.cu sub.cu mul.cu div.cu sum.cu max.cu cuda_Node.cu relu.cu cuda_Tensor.cu nll_loss.cu cuda_Optimizer.cu argmax.cu -lcublas -lcurand -arch=sm_120 -O3

/*
==============================================================================
TODO:
    - Fix naming of cuda versions of functions/classes, i.e., should cuda be
      before or after the original function or class name?
    - SubBackward node's will be created but their backward() is not called,
      and may want to enable grads for `shifted_values = input - max` in
      log_softmax().
    - Addition of bias in matmul() using C=α*AB+β*C for better performance.
    - Multiple Tensor data types.
    - pow() (floats and negatives as exponents).
    - squeeze()/unsqueeze().
    - min() and max() (specifying multiple dimensions to reduce).
    - sum() (specifying multiple dimensions to reduce).
    - mean() (specifying multiple dimensions to reduce).
    - log().
    - cat().
    - == operator.
    - Combine tensor-tensor and tensor-scalar arithmetic operators.
    - Overload the << operator for printing TensorSlice objects, e.g., a row.
    - Modify tensor() to take nested lists of values for creating tensors with
      multiple dimensions, e.g., tensor({{1, 2}, {3, 4}}).

==============================================================================
*/

size_t Time::global_step = 0;

int main() {
    // ---- Load Train Images and Labels from File ----

    cout << "Loading train data..." << endl;
    string train_images_path = "data/images/train_images.bin";
    string train_labels_path = "data/labels/train_labels.bin";
    ifstream train_images_file(train_images_path, ios::binary | ios::ate);
    size_t train_images_file_size = train_images_file.tellg();
    train_images_file.seekg(0, ios::beg); 
    Tensor train_images_host = Tensor::empty({train_images_file_size});
    char byte;
    for (int i = 0; i < train_images_file_size; i++) {
        train_images_file.read(&byte, 1);
        train_images_host[i] = static_cast<float>(static_cast<uint8_t>(byte)) / 255;
    }
    Tensor train_images = Tensor::empty({train_images_file_size}, true);
    cudaMemcpy(train_images.data.get(), train_images_host.data.get(), train_images_file_size * sizeof(float), cudaMemcpyHostToDevice);

    ifstream train_labels_file(train_labels_path, ios::binary | ios::ate);
    size_t train_labels_file_size = train_labels_file.tellg();
    train_labels_file.seekg(0, ios::beg); 
    Tensor train_labels_host = Tensor::empty({train_labels_file_size});
    for (int i = 0; i < train_labels_file_size; i++) {
        train_labels_file.read(&byte, 1);
        train_labels_host[i] = static_cast<float>(static_cast<uint8_t>(byte));
    }
    Tensor train_labels = Tensor::empty({train_labels_file_size}, true);
    cudaMemcpy(train_labels.data.get(), train_labels_host.data.get(), train_labels_file_size * sizeof(float), cudaMemcpyHostToDevice);

    // ---------------------------------------------------------------------------
    // ---- Load Test Images and Labels from File ----

    cout << "Loading test data..." << endl << endl;
    string test_images_path = "data/images/test_images.bin";
    string test_labels_path = "data/labels/test_labels.bin";
    ifstream test_images_file(test_images_path, ios::binary | ios::ate);
    size_t test_images_file_size = test_images_file.tellg();
    test_images_file.seekg(0, ios::beg); 
    Tensor test_images_host = Tensor::empty({test_images_file_size});
    for (int i = 0; i < test_images_file_size; i++) {
        test_images_file.read(&byte, 1);
        test_images_host[i] = static_cast<float>(static_cast<uint8_t>(byte)) / 255;
    }
    Tensor test_images = Tensor::empty({test_images_file_size}, true);
    cudaMemcpy(test_images.data.get(), test_images_host.data.get(), test_images_file_size * sizeof(float), cudaMemcpyHostToDevice);

    ifstream test_labels_file(test_labels_path, ios::binary | ios::ate);
    size_t test_labels_file_size = test_labels_file.tellg();
    test_labels_file.seekg(0, ios::beg); 
    Tensor test_labels_host = Tensor::empty({test_labels_file_size});
    for (int i = 0; i < test_labels_file_size; i++) {
        test_labels_file.read(&byte, 1);
        test_labels_host[i] = static_cast<float>(static_cast<uint8_t>(byte));
    }
    Tensor test_labels = Tensor::empty({test_labels_file_size}, true);
    cudaMemcpy(test_labels.data.get(), test_labels_host.data.get(), test_labels_file_size * sizeof(float), cudaMemcpyHostToDevice);

    // ---------------------------------------------------------------------------
    // ---- Create an CNN Model for MNIST ----

    class Net : public Module {
        Conv2d conv1 = Conv2d(1, 8, {5});
        Conv2d conv2 = Conv2d(8, 16, {5});
        Linear fc1 = Linear(256, 120);
        Linear fc2 = Linear(120, 84);
        Linear fc3 = Linear(84, 10);

        Tensor forward(Tensor& x) override {
            x = relu(conv1(x));
            x = maxpool2d_cuda(x, {2});
            x = relu(conv2(x));
            x = maxpool2d_cuda(x, {2});
            x = x.view({-1, 256});
            x = relu(fc1(x));
            x = relu(fc2(x));
            x = fc3(x);
            return x;
        }
    };

    Net net;

    // ---------------------------------------------------------------------------
    // ---- Model Hyperparameters ----

    size_t epochs = 1;
    size_t batch_size = 64;
    float lr = 0.1;

    // ---------------------------------------------------------------------------
    // ---- Loading Train Data ----

    size_t train_set_count = 60000;
    size_t train_images_pos = 0;
    size_t train_labels_pos = 0;
    vector<Tensor> train_image_batches(train_set_count / batch_size);
    vector<Tensor> train_label_batches(train_set_count / batch_size);
    for (size_t i = 0; i < train_set_count / batch_size; i++) {
        Tensor train_image_batch = Tensor::empty({batch_size, 1, 28, 28}, true);

        cudaMemcpy(train_image_batch.data.get(),
                   train_images.data.get() + train_images_pos,
                   batch_size * 784 * sizeof(float),
                   cudaMemcpyDeviceToDevice);

        train_images_pos += batch_size * 784;
        train_image_batches[i] = train_image_batch;

        Tensor train_label_batch = Tensor::empty({batch_size}, true);

        cudaMemcpy(train_label_batch.data.get(),
                   train_labels.data.get() + train_labels_pos,
                   batch_size * sizeof(float),
                   cudaMemcpyDeviceToDevice);

        train_labels_pos += batch_size;
        train_label_batches[i] = train_label_batch;
    }
    
    // ---------------------------------------------------------------------------
    // ---- Training Loop ----

    Tensor inputs;
    Tensor labels;
    Tensor outputs;

    Tensor loss;
    float loss_value;
    float running_loss = 0;
    Tensor predictions;
    Tensor correct_tensor;
    float correct;
    float accuracy;
    float avg_accuracy = 0;
    
    auto average_duration = 0;
    float total_time = 0;
    float zero_grad_total_time = 0;
    float forward_total_time = 0;
    float loss_total_time = 0;
    float backward_total_time = 0;
    float step_total_time = 0;

    SGD optimizer = SGD(net.parameters(), lr);

    size_t steps = epochs * (train_set_count / batch_size);
    cout << "======== Training... ========" << endl;
    for (size_t step = 0; step < steps; step++) {
        Time::global_step++;
        auto start = chrono::steady_clock::now();
        inputs = train_image_batches[step % (train_set_count / batch_size)];
        labels = train_label_batches[step % (train_set_count / batch_size)];

        auto zero_grad_start = chrono::steady_clock::now();
        optimizer.zero_grad();
        auto zero_grad_stop = chrono::steady_clock::now();
        auto zero_grad_duration = chrono::duration_cast<chrono::microseconds>(zero_grad_stop - zero_grad_start);
        zero_grad_total_time += zero_grad_duration.count();
        auto forward_start = chrono::steady_clock::now();
        outputs = net(inputs);

        auto forward_stop = chrono::steady_clock::now();
        auto forward_duration = chrono::duration_cast<chrono::microseconds>(forward_stop - forward_start);
        forward_total_time += forward_duration.count();

        auto loss_start = chrono::steady_clock::now();
        loss = cross_entropy(outputs, labels);
        auto loss_stop = chrono::steady_clock::now();
        auto loss_duration = chrono::duration_cast<chrono::microseconds>(loss_stop - loss_start);
        loss_total_time += loss_duration.count();

        cudaMemcpy(&loss_value, loss.data.get(), sizeof(float), cudaMemcpyDeviceToHost);
        running_loss += loss_value;

        predictions = outputs.argmax(1);
        correct_tensor = (labels == predictions).sum();
        cudaMemcpy(&correct, correct_tensor.data.get(), sizeof(float), cudaMemcpyDeviceToHost);

        auto backward_start = chrono::steady_clock::now();
        loss.backward();
        auto backward_stop = chrono::steady_clock::now();
        auto backward_duration = chrono::duration_cast<chrono::microseconds>(backward_stop - backward_start);
        backward_total_time += backward_duration.count();

        auto step_start = chrono::steady_clock::now();
        optimizer.step();
        auto step_stop = chrono::steady_clock::now();
        auto step_duration = chrono::duration_cast<chrono::microseconds>(step_stop - step_start);
        step_total_time += step_duration.count();

        accuracy = (correct / batch_size) * 100;
        avg_accuracy += accuracy / 100;
        auto stop = chrono::steady_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
        total_time += duration.count();
        average_duration += duration.count();
        double avg_step_duration_ms = average_duration / 100.0;
        double steps_per_second = 1000.0 / avg_step_duration_ms;
        if (step % 100 == 99) {
            cout << "Step: " << step + 1 << "/" << steps
            << fixed << setprecision(4) << " | Loss: " << running_loss / 100
            << fixed << setprecision(2) << " | Accuracy: " << avg_accuracy << "%"
            << " | Avg Step Duration: " << avg_step_duration_ms << " ms"
            << " | Steps/sec: " << fixed << setprecision(2) << steps_per_second << endl;
            running_loss = 0;
            avg_accuracy = 0;
            average_duration = 0;
        }
    }
    cout << "Overall average step duration: " << total_time / steps << " ms" << endl;
    // cout << "-----------------------------------------------" << endl;
    cout << "Average zero grad duration: " << zero_grad_total_time / steps << " μs" << endl;
    cout << "Average forward duration: " << forward_total_time / steps << " μs" << endl;
    cout << "Average loss duration: " << loss_total_time / steps << " μs" << endl;
    cout << "Average backward duration: " << backward_total_time / steps << " μs" << endl;
    cout << "Average step duration: " << step_total_time / steps << " μs" << endl;

    // // ---------------------------------------------------------------------------
    // // ---- Loading Test Data ----

    size_t test_set_count = 10000;
    size_t test_images_pos = 0;
    size_t test_labels_pos = 0;
    float total_correct = 0;
    vector<Tensor> test_image_batches(test_set_count / batch_size);
    vector<Tensor> test_label_batches(test_set_count / batch_size);
    for (size_t i = 0; i < test_set_count / batch_size; i++) {
        Tensor test_image_batch = Tensor::empty({batch_size, 1, 28, 28}, true);

        cudaMemcpy(test_image_batch.data.get(),
                   test_images.data.get() + test_images_pos,
                   batch_size * 784 * sizeof(float),
                   cudaMemcpyDeviceToDevice);

        test_images_pos += batch_size * 784;
        test_image_batches[i] = test_image_batch;

        Tensor test_label_batch = Tensor::empty({batch_size}, true);
        
        cudaMemcpy(test_label_batch.data.get(),
                   test_labels.data.get() + test_labels_pos,
                   batch_size * sizeof(float),
                   cudaMemcpyDeviceToDevice);

        test_labels_pos += batch_size;
        test_label_batches[i] = test_label_batch;
    }

    // ---------------------------------------------------------------------------
    // ---- Evaluation Loop ----

    steps = test_set_count / batch_size;
    float batch_loss = 0;
    running_loss = 0;
    correct = 0;
    cout << "======== Evaluating... ========" << endl;
    for (size_t step = 0; step < steps; step++) {
        inputs = test_image_batches[step % (test_set_count / batch_size)];
        labels = test_label_batches[step % (test_set_count / batch_size)];

        outputs = net(inputs);
        loss = cross_entropy(outputs, labels);

        cudaMemcpy(&loss_value, loss.data.get(), sizeof(float), cudaMemcpyDeviceToHost);
        batch_loss += loss_value;
        running_loss += loss_value;

        predictions = outputs.argmax(1);
        correct_tensor = (labels == predictions).sum();
        cudaMemcpy(&correct, correct_tensor.data.get(), sizeof(float), cudaMemcpyDeviceToHost);
        total_correct += correct;

        if (step % 10 == 9) {
            cout << "Step: " << step + 1 << "/" << steps << fixed << setprecision(4) << " | Loss: " << batch_loss / 10 << endl;
            batch_loss = 0;
        }

        Engine::clear_graph(loss.node);
    }
    accuracy = (total_correct / test_set_count) * 100;
    float avg_loss = running_loss / steps;
    cout << endl << fixed << setprecision(2) << "Test Accuracy: " << accuracy << "%" << " | Average Loss: " << fixed << setprecision(4) << avg_loss << endl;
}
