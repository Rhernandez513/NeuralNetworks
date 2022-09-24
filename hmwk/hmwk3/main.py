import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST


def u(x) -> np.ndarray:
    """Step function applied component wise"""
    return np.array([1.0 if x_i >= 0 else 0.0 for x_i in x])

def run_test(W, n, test_images, test_labels, section_str) -> None:
    errors = 0
    for i in range(n):
        # calculate the induced local fields v
        x_i = np.array(test_images[i])
        v = np.dot(W, x_i)
        largest_component = np.argmax(v)
        if largest_component != test_labels[i]:
            errors += 1
    print("section {} finished".format(section_str))
    print("run of testing data with n = {}".format(n))
    print("error rate: {error_rate}".format(error_rate=errors/n))
    print("error count: {error_count}".format(error_count=errors))

def run_train(eta, epsilon, n, epoch_max, training_images, training_labels, section_str) -> np.ndarray:
    # section d.1
    W = np.random.uniform(-1, 1, size=(10, 784)) # weight matrix

    # section d.2
    epoch = 0

    # section d.3
    errors = [0 for i in range(epoch_max)]

    # so much easier to just use numpy and matrix multiplication

    # section d.3.1
    while epoch < epoch_max:
        for i in range(n):
            x_i = np.array(training_images[i])
            label = training_labels[i]

            # v=Wx aka induced local field
            # we do not add any biases for this case
            v = np.dot(W, x_i)
            predicted_label = np.argmax(v)
            if predicted_label != label:
                errors[epoch] += 1
        epoch += 1
        for i in range(n):
            x_i = np.array(training_images[i])
            x_i.resize(784, 1)

            label = training_labels[i]
            d_i = np.zeros((10,1))
            d_i[training_labels[i]] = 1

            # v=W * x_i aka induced local fields
            v = np.dot(W, x_i)

            y = u(v)
            y.resize(10, 1)

            # we have to resize both d_i and y to be 10x1
            # then we can mult by x_i.T to get a 10x784 matrix

            W += eta * (d_i - y) * x_i.T

        if (errors[epoch - 1]/n) > epsilon:
            # section 3.2
            break

    print(section_str)
    print("finished training after {epoch} epochs".format(epoch=epoch))
    print("error rate: {error_rate}".format(error_rate=errors[epoch - 1]/n))
    print("error count: {error_count}".format(error_count=errors[epoch - 1]))

    # We will avoid polluting the output with the weight matrix
    # print("\nW: {W}\n".format(W=W))

    return W

def main():

    mndata = MNIST('resources')

    # each image of the images list is a python list of unsigned bytes
    # In the images: "0" is white, "255" is black, we see in the python-mnist source code library that >200 is dark enough to be considered writing
    # each image is 28x28 pixels
    # labels is a python array of insigned bytes, luckily python translates these to ints no problem
    training_images, training_labels = mndata.load_training()


    # section d.0
    eta = 0.1
    epsilon = 0.0
    n = 1000
    epoch_max = 100
    W = run_train(eta, epsilon, n, epoch_max, training_images, training_labels, "Section A through D")

    # section e.0
    # Given the same W matrix from training
    testing_images, testing_labels = mndata.load_testing()
    # section e.1 && e.2
    n = 1000
    run_test(W, n, testing_images, testing_labels, "E")

    # section f
    # n = 50
    # eta = 1
    # episilon = 0.0
    # print("section F, testing data with n = {}".format(n), end=" ")
    # print("and eta = {}".format(eta), end=" ")
    # print("and epsilon = {}".format(episilon))



if __name__ == '__main__':
    main()


# EOF
