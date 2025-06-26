# CNN-in-CPP
A deep learning framework in C++ similar to PyTorch, originally built for CNNs but now supporting general neural network development.

# Overview

What began as a simple CNN implementation in C++ evolved into a full-fledged deep learning framework inspired by PyTorch. I built it to test my understanding of deep learning, challenge my assumptions, and deepen my knowledge of C++ and CUDA, which form the GPU-accelerated backends of deep learning frameworks like PyTorch and TensorFlow.

With a familiar PyTorch-style syntax, you can define and train efficient neural networks that achieve over 99% test accuracy in just a few lines of C++. The framework is optimized for performance and can run entirely on the GPU. In many cases, simple models run at speeds comparable to their PyTorch counterparts.

While the current focus is on CNNs, I plan to expand the framework to support other domains such as natural language processing (NLP).

# Example
Create a CNN model for classifying handwritten digits from the MNIST dataset, inspired
by LeNet-5, similar to [this example](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#define-a-convolutional-neural-network) in PyTorch.
```python
class Net : public Module {
        Conv2d conv1 = Conv2d(1, 6, {5});
        MaxPool2d pool = MaxPool2d({2});
        Conv2d conv2 = Conv2d(6, 16, {5});
        Linear fc1 = Linear(256, 120);
        Linear fc2 = Linear(120, 84);
        Linear fc3 = Linear(84, 10);

        Tensor forward(Tensor& x) override {
            x = pool(relu(conv1(x)));
            x = pool(relu(conv2(x)));
            x = x.view({-1, 256});
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
