import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST


def PerceptronNode():
    def __init(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def get_output(self, inputs):
        return np.dot(self.weights, inputs) + self.bias

    def get_weights(self):
        return self.weights
    
    def get_bias(self):
        return self.bias

    def set_weights(self, weights):
        self.weights = weights
    
    def set_bias(self, bias):
        self.bias = bias

def PerceptronLayer():
    def __init__(self, num_nodes, num_inputs):
        self.num_nodes = num_nodes
        self.num_inputs = num_inputs
        self.nodes = [PerceptronNode() for i in range(num_nodes)]
        for node in self.nodes:
            node.set_weights(np.random.rand(num_inputs))
            node.set_bias(np.random.rand())

    def get_output(self, inputs):
        return [node.get_output(inputs) for node in self.nodes]

    def get_weights(self):
        return [node.get_weights() for node in self.nodes]

    def get_bias(self):
        return [node.get_bias() for node in self.nodes]

    def set_weights(self, weights):
        for i in range(self.num_nodes):
            self.nodes[i].set_weights(weights[i])

    def set_bias(self, bias):
        for i in range(self.num_nodes):
            self.nodes[i].set_bias(bias[i])


def PerceptronNetwork():
    def __init__(self, num_layers, num_nodes, num_inputs):
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.num_inputs = num_inputs
        self.layers = [PerceptronLayer(num_nodes, num_inputs) for i in range(num_layers)]
        for layer in self.layers:
            layer.set_weights(np.random.rand(num_nodes, num_inputs))
            layer.set_bias(np.random.rand(num_nodes))

    def get_output(self, inputs):
        return [layer.get_output(inputs) for layer in self.layers]

    def get_weights(self):
        return [layer.get_weights() for layer in self.layers]

    def get_bias(self):
        return [layer.get_bias() for layer in self.layers]

    def set_weights(self, weights):
        for i in range(self.num_layers):
            self.layers[i].set_weights(weights[i])

    def set_bias(self, bias):
        for i in range(self.num_layers):
            self.layers[i].set_bias(bias[i])


def main():

    mndata = MNIST('resources')

    # each image of the images list is a python list of unsigned bytes
    # In the images: "0" is white, "255" is black, we see in the python-mnist source code library that >200 is dark enough to be considered writing
    # each image is 28x28 pixels

    # labels is a python array of insigned bytes, luckily python translates these to ints no problem
    images, labels = mndata.load_training()

    # testing_images, testing_labels = mndata.load_testing()

    # should render a 2 
    print(mndata.display(images[5]))
    # should print 2 to console
    print(labels[5])


if __name__ == '__main__':
    main()


# EOF
