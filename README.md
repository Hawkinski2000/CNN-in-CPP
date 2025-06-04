# CNN-in-CPP
A convolutional neural network (CNN) implemented in C++.

Built with modularity in mind, this project follows a PyTorch-like structure to simplify model building and training.

# Example
Create an MLP model for classifying handwritten digits from the MNIST dataset:
```
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
```
Training is as simple as:
```
SGD optimizer = SGD(net.parameters(), lr);

for (size_t step = 0; step < steps; step++) {
        inputs = train_image_batches[step % (train_set_count / batch_size)];
        labels = train_label_batches[step % (train_set_count / batch_size)];

        optimizer.zero_grad();

        outputs = net(inputs);
        loss = cross_entropy(outputs, labels);
        loss.backward();
        optimizer.step();
    }
```