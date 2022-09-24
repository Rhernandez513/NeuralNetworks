import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST


def u(x) -> np.ndarray:
    """Step function applied component wise"""
    return np.array([1.0 if x_i >= 0 else 0.0 for x_i in x])

def run_test(W, n, test_images, test_labels, section_str) -> tuple:
    error_count = 0
    for i in range(n):
        # calculate the induced local fields v
        x_i = np.array(test_images[i])
        v = np.dot(W, x_i)
        largest_component = np.argmax(v)
        if largest_component != test_labels[i]:
            error_count += 1
    error_rate = error_count/n
    print("Section {}".format(section_str))
    print("run of TEST data ")
    print("n: {}".format(n))
    print("error count: {error_count}".format(error_count=error_count))
    print("Percent of TEST errors: {error_rate}".format(error_rate=error_rate))
    print()
    return (error_rate, error_count)

def run_train(eta, epsilon, n, epoch_max, training_images, training_labels, section_str) -> tuple:
    print("Training with eta=" + str(eta) + " n=" + str(n) + " epsilon=" + str(epsilon))
    # section d.1
    W = np.random.uniform(-1, 1, size=(10, 784)) # weight matrix

    # section d.2
    epoch = 0

    # section d.3
    errors = np.zeros(epoch_max)

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

        # section 3.2
        if (errors[epoch - 1]/n) > epsilon:
            break

    print("Section {}".format(section_str))
    print("finished TRAINING after {epoch} epoch(s)".format(epoch=epoch))
    print("eta: {eta}".format(eta=eta))
    print("epsilon: {epsilon}".format(epsilon=epsilon))
    print("n: {n}".format(n=n))
    print("epoch_max: {epoch_max}".format(epoch_max=epoch_max))
    print("Percent errors: {error_rate}".format(error_rate=errors[epoch - 1]/n))
    print("error count: {error_count}".format(error_count=errors[epoch - 1]))
    print()

    # We will avoid polluting the output with the weight matrix
    # print("\nW: {W}\n".format(W=W))

    return (W, errors)

def main():

    mndata = MNIST('resources')

    # each image of the images list is a python list of unsigned bytes
    # In the images: "0" is white, "255" is black, we see in the python-mnist source code library that >200 is dark enough to be considered writing
    # each image is 28x28 pixels
    # labels is a python array of insigned bytes, luckily python translates these to ints no problem
    training_images, training_labels = mndata.load_training()
    testing_images, testing_labels = mndata.load_testing()


    # section d.0
    eta = 0.1
    epsilon = 0.1
    n = 1000
    epoch_max = 100
    (W, errors) = run_train(eta, epsilon, n, epoch_max, training_images, training_labels, "A through D")

    # section e.0
    # Given the same W matrix from training
    # section e.1 && e.2
    n = 1000
    (error_rate, error_count) = run_test(W, n, testing_images, testing_labels, "E")

    # section f
    n = 50
    eta = 1
    epsilon = 0.01
    (W, errors) = run_train(eta, epsilon, n, epoch_max, training_images, training_labels, "F")

    plt.title("Error Rate vs Epoch, N={}, Eta={}, Epsilon={}".format(n, eta, epsilon))
    plt.xlabel("Epoch")
    plt.ylabel("Error Rate")
    plt.plot(errors, [i for i in range(epoch_max)])
    plt.show()

    n = 1000
    (error_rate, error_count) = run_test(W, n, testing_images, testing_labels, "F")

    # section g
    n = 1000
    eta = 1
    epsilon = 0.01
    (W, errors) = run_train(eta, epsilon, n, epoch_max, training_images, training_labels, "G")

    plt.title("Error Rate vs Epoch, N={}, Eta={}, Epsilon={}".format(n, eta, epsilon))
    plt.xlabel("Epoch")
    plt.ylabel("Error Rate")
    plt.plot(errors, [i for i in range(epoch_max)])
    plt.show()

    n = 10000
    (error_rate, error_count) = run_test(W, n, testing_images, testing_labels, "G")

    # section h
    n = 60000
    epsilon = 0.0
    (W, errors) = run_train(eta, epsilon, n, epoch_max, training_images, training_labels, "H")

    plt.title("Error Rate vs Epoch, N={}, Eta={}, Epsilon={}".format(n, eta, epsilon))
    plt.xlabel("Epoch")
    plt.ylabel("Error Rate")
    plt.plot(errors, [i for i in range(epoch_max)])
    plt.show()

    n = 10000
    (error_rate, error_count) = run_test(W, n, testing_images, testing_labels, "H")

    # section i
    epsilon = 0.1
    eta = 0.1
    n = 60000
    (W, errors) = run_train(eta, epsilon, n, epoch_max, training_images, training_labels, "I")
    plt.title("Error Rate vs Epoch, N={}, Eta={}, Epsilon={}".format(n, eta, epsilon))
    plt.xlabel("Epoch")
    plt.ylabel("Error Rate")
    plt.plot(errors, [i for i in range(epoch_max)])
    plt.show()
    n = 10000
    (error_rate, error_count) = run_test(W, n, testing_images, testing_labels, "I")

if __name__ == '__main__':
    main()


# EOF
