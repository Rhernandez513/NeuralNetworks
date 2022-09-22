import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST


class PerceptronNode():
    def __init(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def get_output(self, inputs):
        return np.dot(self.weights, inputs) + self.bias
    
    def set_weights(self, weights):
        self.weights = weights


class PerceptronLayer():
    def __init__(self, num_nodes, num_inputs):
        self.num_nodes = num_nodes
        self.num_inputs = num_inputs
        self.nodes = [PerceptronNode() for i in range(num_nodes)]
        for node in self.nodes:
            node.set_weights(np.random.rand(num_inputs))
            # for this case we will ignore the bias
            node.bias = 0


class PerceptronNetwork():
    def __init__(self, num_layers, num_input_nodes, num_inputs, num_outputs):
        self.num_layers = num_layers
        self.num_nodes = num_input_nodes
        self.num_inputs = num_inputs

        # we actually want a network with 784 input nodes and 10 output nodes for this specific case
        layer_one = PerceptronLayer(num_input_nodes, num_inputs)
        layer_two = PerceptronLayer(num_outputs, num_inputs)
        self.layers = [layer_one, layer_two]


def step_function(x) -> int:
    return 1 if x >= 0 else 0


def arr_step_function(x) -> np.ndarray:
    return np.array([step_function(x_i) for x_i in x])


def main():

    mndata = MNIST('resources')

    # each image of the images list is a python list of unsigned bytes
    # In the images: "0" is white, "255" is black, we see in the python-mnist source code library that >200 is dark enough to be considered writing
    # each image is 28x28 pixels

    # labels is a python array of insigned bytes, luckily python translates these to ints no problem
    images, labels = mndata.load_training()

    # testing_images, testing_labels = mndata.load_testing()

    # num_layers = 2
    # num_input_nodes, num_inputs = 784, 784
    # num_outputs = 10
    # perceptron_network = PerceptronNetwork(num_layers, num_input_nodes, num_inputs, num_outputs)


    # section d.0
    eta = 0.1
    episilon = 0.0
    n = 1000

    # section d.1
    W = np.random.uniform(-1, 1, size=(10, 784)) # weight matrix

    # section d.2
    epoch = 0
    epoch_max = 100

    # section d.3
    errors = [0 for i in range(epoch_max)]

    # so much easier to just use numpy and matrix multiplication

    # section d.3.1
    while epoch < epoch_max:
        for i in range(n):
            x = images[i]
            d_i = labels[i]
            # v=Wx aka induced local field
            # we do not add any biases for this case
            v = np.dot(W, x)
            prediction = np.argmax(v)
            if prediction != d_i:
                errors[epoch] += 1
        epoch += 1
        for i in range(n):
            x = images[i]
            d_i = labels[i]
            v = np.dot(W, x)
            prediction = np.argmax(v)
            v = [1 if i == prediction else 0 for i in range(10)]
            y = arr_step_function(v)
            W += eta * (d_i - y) * x.T
            # W = W + eta * (d_i - y) * x.T
            pass
        if (errors[epoch - 1]/n) > episilon:
            # section 3.2
            break




    #     pass


    # should render a 2 
    print(mndata.display(images[5]))
    # should print 2 to console
    print(labels[5])


if __name__ == '__main__':
    main()


# EOF
