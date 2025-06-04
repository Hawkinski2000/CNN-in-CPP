# CNN-in-CPP
A convolutional neural network (CNN) implemented in C++.

Built with modularity in mind, this project follows a PyTorch-like structure to simplify model building and training.

# Example
```
// ---- Create an MLP Model for MNIST ----

    class Net : public Module {
        public:
            Linear fc1 = Linear(784, 512);
            Linear fc2 = Linear(512, 256);
            Linear fc3 = Linear(256, 10);
            
            Net() {
                modules = {&fc1, &fc2, &fc3};
            }

            Tensor forward(Tensor& x) override {
                x = relu(fc1(x));
                x = relu(fc2(x));
                x = fc3(x);
                return x;
            }
    };

    Net net;
```
