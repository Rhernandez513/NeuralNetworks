import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST


def u(x) -> np.ndarray:
    """Step function applied component wise"""
    return np.array([1.0 if x_i >= 0 else 0.0 for x_i in x])

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
            x_i = np.array(images[i])
            label = labels[i]

            # v=Wx aka induced local field
            # we do not add any biases for this case
            v = np.dot(W, x_i)
            predicted_label = np.argmax(v)
            if predicted_label != label:
                errors[epoch] += 1
        epoch += 1
        for i in range(n):
            x_i = np.array(images[i])
            x_i.resize(784, 1)

            label = labels[i]
            d_i = np.zeros((10,1))
            d_i[labels[i]] = 1

            # v=W * x_i aka induced local fields
            v = np.dot(W, x_i)

            y = u(v)
            y.resize(10, 1)

            # we have to resize both d_i and y to be 10x1
            # then we can mult by x_i.T to get a 10x784 matrix

            W += eta * (d_i - y) * x_i.T

        if (errors[epoch - 1]/n) > episilon:
            # section 3.2
            break

    print("finished training after {epoch} epochs".format(epoch=epoch))
    print("error rate: {error_rate}".format(error_rate=errors[epoch - 1]/n))
    print("error count: {error_count}".format(error_count=errors[epoch - 1]))
    print("W: {W}".format(W=W))


if __name__ == '__main__':
    main()


# EOF
