#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def get_d_vector(x, v, n) -> np.ndarray:
    return np.array(
        [np.sin(20 * x[i]) + (3 * x[i]) + v[i] for i in range(n)]
    )


def output_phi(v):
    return v


def phi(v):
    return np.tanh(v)


def phi_prime(v):
    """derivative of tanh(v) is sec^2(v) aka (1/cosh^2(v))
    https://www.wolframalpha.com/input?i=derivative+of+tanh%28x%29
    """
    return 1 / (np.cosh(v) ** 2)


def output_phi_prime(v):
    """dv of v is 1"""
    return 1


def main():
    # Notes: We will use a neural network for curve fitting.
    # section 1
    n = 300
    x_1_n = np.random.uniform(0, 1, n)
    # section 2
    v_1_n = np.random.uniform((-1/10), (1/10), n)
    # section 3
    d_i_n = get_d_vector(x_1_n, v_1_n, n)
    plt.title('x v d, n=' + str(n))
    plt.scatter(x_1_n, d_i_n)
    # TODO uncomment for report
    plt.show()

    # Notes
    # We will use a neural network with 1 input
    # 1 hidden layer with 24 neurons
    # 1 output layer with 1 neuron
    # begin with a bias of 1

    # for weights from input to hidden layer we have (N weights + N bias)
    # for the weights from the hidden layer to output layer we have (N weights + 1 bias)
    # thus we have 3N+1 weights
    W = np.random.uniform(-1, 1, size=(2,24)) # weight matrix
    input_biases = np.array([1 for i in range(24)])
    W = np.vstack((input_biases, W)) # we will use W[0] for input biases
    output_bias = 1


    # Section 4 start on the backpropagation algorithm
    eta = 1.0
    epsilon = 1.0

    # from lecture notes
    # 1. Init weights randomly
    # 2. For epochs 1...n
    #   2.1 for i to |S|
    #   2.1.1 w <- w + eta(d_i - phi(w.T * x_i)) * phi'(w.T * x_i) * x_i
    # Note here we use biases, so we add b as the first column of W

    epoch = 0
    epoch_max = 100
    errors = np.zeros(epoch_max)
    mse = float('inf')
    mean_square_errors = np.zeros((epoch_max, n))
    while epoch < epoch_max:
        for i in range(n):
            x_i = x_1_n[i]
            d_i = d_i_n[i]
            v = np.dot(W, x_i) # local fields of first layer
            y_1 = phi(v) # output of first layer
            y_2 = output_phi(output_bias + np.sum(W[2])) # output of second layer

            if y_2 != d_i:
                errors[epoch] += 1

                # weight update with backpropagation algo
                W += eta * (d_i - phi(v)) * phi_prime(v) * x_i
                output_bias = W[0][0] # update output bias too ?

            # Mean Square Error
            new_mse = np.mean((d_i_n - y_2)) ** 2
            mean_square_errors[epoch][i] = new_mse
            if new_mse > mse:
                # we are increasing the MSE, so we need to modify eta
                eta *= 0.9
            mse = new_mse

        if (errors[epoch - 1]/n) > epsilon or abs(mean_square_errors[epoch][i-1] - mean_square_errors[epoch][i]) < 0.0001:
            break
        epoch += 1

    print("final mse: {}".format(mse))
    print("Epochs: {}".format(epoch))
    print("Errors: {}".format(errors))
    print(W)
    print()

    # TODO plot epochs vs mean squared error


if __name__ == '__main__':
    main()

# EOF
