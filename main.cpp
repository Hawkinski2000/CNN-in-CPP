#include <iostream>
#include <fstream>
#include <iomanip>
#include "Tensor.h"
#include "Engine.h"
#include "nn.h"
#include "functional.h"
#include "optim.h"
using namespace std;


// Compile with: nvcc -o main main.cpp Tensor.cpp Node.cpp Engine.cpp nn.cpp functional.cpp optim.cpp matmul.cu -lcublas

/*
==============================================================================
TODO:
    - Multiple Tensor data types.
    - pow() (floats and negatives as exponents).
    - squeeze()/unsqueeze().
    - min() and max() (specifying dimensions to reduce).
    - sum() (specifying multiple dimensions to reduce).
    - mean() (specifying dimensions to reduce).
    - exp().
    - log().
    - cat().
    - == operator.
    - Combine tensor-tensor and tensor-scalar arithmetic operators.
    - Overload the << operator for printing TensorSlice objects, e.g., a row.
    - Modify tensor() to take nested lists of values for creating tensors with
      multiple dimensions, e.g., tensor({{1, 2}, {3, 4}}).

==============================================================================
*/


int main() {  
// ---- Load Train Images and Labels from File ----

    string train_images_path = "data/images/train_images.bin";
    string train_labels_path = "data/labels/train_labels.bin";
    ifstream train_images_file(train_images_path, ios::binary | ios::ate);
    size_t train_images_file_size = train_images_file.tellg();
    train_images_file.seekg(0, ios::beg); 
    Tensor train_images = Tensor::empty({train_images_file_size});
    char byte;
    for (int i = 0; i < train_images_file_size; i++) {
        train_images_file.read(&byte, 1);
        train_images[i] = static_cast<float>(static_cast<uint8_t>(byte)) / 255;
    }
    ifstream train_labels_file(train_labels_path, ios::binary | ios::ate);
    size_t train_labels_file_size = train_labels_file.tellg();
    train_labels_file.seekg(0, ios::beg); 
    Tensor train_labels = Tensor::empty({train_labels_file_size});
    for (int i = 0; i < train_labels_file_size; i++) {
        train_labels_file.read(&byte, 1);
        train_labels[i] = static_cast<float>(static_cast<uint8_t>(byte));
    }

    // ---------------------------------------------------------------------------
    // ---- Load Test Images and Labels from File ----

    string test_images_path = "data/images/test_images.bin";
    string test_labels_path = "data/labels/test_labels.bin";
    ifstream test_images_file(test_images_path, ios::binary | ios::ate);
    size_t test_images_file_size = test_images_file.tellg();
    test_images_file.seekg(0, ios::beg); 
    Tensor test_images = Tensor::empty({test_images_file_size});
    byte;
    for (int i = 0; i < test_images_file_size; i++) {
        test_images_file.read(&byte, 1);
        test_images[i] = static_cast<float>(static_cast<uint8_t>(byte)) / 255;
    }
    ifstream test_labels_file(test_labels_path, ios::binary | ios::ate);
    size_t test_labels_file_size = test_labels_file.tellg();
    test_labels_file.seekg(0, ios::beg); 
    Tensor test_labels = Tensor::empty({test_labels_file_size});
    for (int i = 0; i < test_labels_file_size; i++) {
        test_labels_file.read(&byte, 1);
        test_labels[i] = static_cast<float>(static_cast<uint8_t>(byte));
    }

    // ---------------------------------------------------------------------------
    // ---- Create an MLP Model for MNIST ----

    class Net : public Module {
        Linear fc1 = Linear(784, 512);
        Linear fc2 = Linear(512, 256);
        Linear fc3 = Linear(256, 10);

        Tensor forward(Tensor& x) override {
            x = relu(fc1(x));
            x = relu(fc2(x));
            x = fc3(x);
            return x;
        }
    };

    Net net;

    // ---------------------------------------------------------------------------
    // ---- Model Hyperparameters ----

    size_t epochs = 5;
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
        Tensor train_image_batch = Tensor::empty({batch_size, 784});
        for (size_t k = 0; k < batch_size * 784; k++) {
            train_image_batch.data.get()[k] = train_images.data.get()[train_images_pos + k];
        }
        train_images_pos += batch_size * 784;
        train_image_batches[i] = train_image_batch;

        Tensor train_label_batch = Tensor::empty({batch_size});
        for (size_t k = 0; k < batch_size; k++) {
            train_label_batch.data.get()[k] = train_labels.data.get()[train_labels_pos + k];
        }
        train_labels_pos += batch_size;
        train_label_batches[i] = train_label_batch;
    }

    // ---------------------------------------------------------------------------
    // ---- Training Loop ----

    Tensor inputs;
    Tensor labels;
    Tensor outputs;
    Tensor loss;
    Tensor predictions;
    float running_loss = 0;
    float accuracy;
    float avg_accuracy = 0;
    float total = batch_size;
    float correct;

    SGD optimizer = SGD(net.parameters(), lr);

    size_t steps = epochs * (train_set_count / batch_size);
    cout << "======== Training... ========" << endl;
    for (size_t step = 0; step < steps; step++) {
        inputs = train_image_batches[step % (train_set_count / batch_size)];
        labels = train_label_batches[step % (train_set_count / batch_size)];

        optimizer.zero_grad();

        outputs = net(inputs);
        loss = cross_entropy(outputs, labels);
        loss.backward();
        optimizer.step();

        running_loss += loss[0];
        predictions = outputs.argmax(1);
        correct = 0;
        for (size_t j = 0; j < predictions.numel(); j++) {
            if (labels.data.get()[j] == predictions.data.get()[j]) {
                correct += 1;
            }
        }
        accuracy = (correct / total) * 100;
        avg_accuracy += accuracy / 100;
        if (step % 100 == 99) {
            cout << "Step: " << step + 1 << "/" << steps << fixed << setprecision(4) << " | Loss: " << running_loss / 100 << fixed << setprecision(2) << " | Accuracy: " << avg_accuracy << "%" << endl;
            running_loss = 0;
            avg_accuracy = 0;
        }
    }

    // ---------------------------------------------------------------------------
    // ---- Loading Test Data ----

    size_t test_set_count = 10000;
    size_t test_images_pos = 0;
    size_t test_labels_pos = 0;
    vector<Tensor> test_image_batches(test_set_count / batch_size);
    vector<Tensor> test_label_batches(test_set_count / batch_size);
    for (size_t i = 0; i < test_set_count / batch_size; i++) {
        Tensor test_image_batch = Tensor::empty({batch_size, 784});
        for (size_t k = 0; k < batch_size * 784; k++) {
            test_image_batch.data.get()[k] = test_images.data.get()[test_images_pos + k];
        }
        test_images_pos += batch_size * 784;
        test_image_batches[i] = test_image_batch;

        Tensor test_label_batch = Tensor::empty({batch_size});
        for (size_t k = 0; k < batch_size; k++) {
            test_label_batch.data.get()[k] = test_labels.data.get()[test_labels_pos + k];
        }
        test_labels_pos += batch_size;
        test_label_batches[i] = test_label_batch;
    }

    // ---------------------------------------------------------------------------
    // ---- Evaluation Loop ----

    steps = test_set_count / batch_size;
    running_loss = 0;
    correct = 0;
    float avg_loss = 0;
    cout << "======== Evaluating... ========" << endl;
    for (size_t step = 0; step < steps; step++) {
        inputs = test_image_batches[step % (test_set_count / batch_size)];
        labels = test_label_batches[step % (test_set_count / batch_size)];

        outputs = net(inputs);
        loss = cross_entropy(outputs, labels);

        running_loss += loss[0];
        predictions = outputs.argmax(1);
        for (size_t j = 0; j < predictions.numel(); j++) {
            if (labels.data.get()[j] == predictions.data.get()[j]) {
                correct += 1;
            }
        }
        if (step % 10 == 9) {
            cout << "Step: " << step + 1 << "/" << steps << fixed << setprecision(4) << " | Loss: " << running_loss / 10 << endl;
            running_loss = 0;
        }
        avg_loss += loss.get_data()[0] / steps;

        Engine::clear_graph(loss.node);
    }
    accuracy = (correct / test_set_count) * 100;
    cout << endl << fixed << setprecision(2) << "Test Accuracy: " << accuracy << "%" << " | Average Loss: " << fixed << setprecision(4) << avg_loss << endl;
}