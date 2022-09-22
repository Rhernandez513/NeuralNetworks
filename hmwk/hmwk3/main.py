from tkinter import W
import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST


class PerceptronNode():
    def __init(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def get_output(self, inputs):
        return np.dot(self.weights, inputs) + self.bias

class PerceptronLayer():
    def __init__(self, num_nodes, num_inputs):
        self.num_nodes = num_nodes
        self.num_inputs = num_inputs
        self.nodes = [PerceptronNode() for i in range(num_nodes)]
        for node in self.nodes:
            node.set_weights(np.random.rand(num_inputs))
            # for this case we will ignore the bias
            # node.set_bias(np.random.rand())
            node.set_bias(0)


class PerceptronNetwork():
    def __init__(self, num_layers, num_input_nodes, num_inputs, num_outputs):
        self.num_layers = num_layers
        self.num_nodes = num_input_nodes
        self.num_inputs = num_inputs

        # we actually want a network with 784 input nodes and 10 output nodes for this specific case
        layer_one = PerceptronLayer(num_input_nodes, num_inputs)
        layer_two = PerceptronLayer(num_outputs, num_inputs)
        self.layers = [layer_one, layer_two]

        # we actually init weights in the PerceptronLayer constructor
        # self.layers = [PerceptronLayer(num_nodes, num_inputs) for i in range(num_layers)]
        # for layer in self.layers:
        #     layer.set_weights(np.random.rand(num_nodes, num_inputs))
        #     layer.set_bias(np.random.rand(num_nodes))


def step_function(x) -> int:
    return 1 if x >= 0 else 0


def arr_step_function(x) -> np.ndarray:
    return np.array([step_function(xi) for xi in x])


def main():

    mndata = MNIST('resources')

    # each image of the images list is a python list of unsigned bytes
    # In the images: "0" is white, "255" is black, we see in the python-mnist source code library that >200 is dark enough to be considered writing
    # each image is 28x28 pixels

    # labels is a python array of insigned bytes, luckily python translates these to ints no problem
    images, labels = mndata.load_training()

    # testing_images, testing_labels = mndata.load_testing()

    num_layers = 2
    num_input_nodes, num_inputs = 784, 784
    num_outputs = 10
    perceptron_network = PerceptronNetwork(num_layers, num_input_nodes, num_inputs, num_outputs)

    w = np.random.uniform(-1, 1, size=(10, 784))
    epoch = 0
    eta = 0.1
    episilon = 0.0
    n = 1000

    # should render a 2 
    print(mndata.display(images[5]))
    # should print 2 to console
    print(labels[5])


if __name__ == '__main__':
    main()


# EOF
