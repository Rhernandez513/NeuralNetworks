#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def get_d_vector(x, v, n) -> np.ndarray:
    return np.array(
        [np.sin(20 * x[i]) + (3 * x[i]) + v[i] for i in range(n)]
    )


def output_phi(u):
    return u


def phi(v):
    return np.tanh(v)


def phi_prime(v):
    """derivative of tanh(v) is sec^2(v) aka (1/cosh^2(v))
    https://www.wolframalpha.com/input?i=derivative+of+tanh%28x%29
    """
    return 1 / (np.cosh(v) ** 2)


def output_phi_prime(u):
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
    plt.show()

    # Notes
    # We will use a neural network with 1 input
    # 1 hidden layer with 24 neurons
    # 1 output layer with 1 neuron
    # begin with a bias of 1

    # for weights from input to hidden layer we have (N weights + N bias)
    # for the weights from the hidden layer to output layer we have (N weights + 1 bias)
    # thus we have 3N+1 weights
    W_1 = np.random.uniform(-1, 1, size=(1,24)) # weight matrix for input to hidden layer
    W_2 = np.random.uniform(-1, 1, size=(1,24)) # weight matrix for hidden to output layer
    input_biases = np.array([1 for i in range(24)])
    W_1 = np.vstack((input_biases, W_1)) # we will use W_1[0] for input biases
    output_bias = 1


    # Section 4 start on the backpropagation algorithm
    eta = 0.1
    epsilon = 0.9

    # from lecture notes
    # 1. Init weights randomly
    # 2. For epochs 1...n
    #   2.1 for i to |S|
    #   2.1.1 w <- w + eta(d_i - phi(w.T * x_i)) * phi'(w.T * x_i) * x_i
    # Note here we use biases, so we add b as the first column of W

    epoch = 0
    epoch_max = 1000
    errors = np.zeros(epoch_max)
    mse = float('inf')
    mean_square_errors = np.zeros((epoch_max, n))
    while epoch < epoch_max:
        for i in range(n):
            x_i = x_1_n[i]
            d_i = d_i_n[i]

            # forward pass
            v = np.add(W_1[0], W_1[1] * x_i) # local field of first layer, W[0] are biases
            y_1 = phi(v) # output of first layer
            u = (W_2 * y_1) + output_bias # local field of second layer
            y_2 = output_phi(np.sum(u)) # output of second layer

            # last layer error
            l2_error = d_i - y_2

            if abs(l2_error) > epsilon:
                errors[epoch] += 1

                # weight update with backpropagation algo
                output_bias += eta * l2_error # update output bias
                W_2 += eta * l2_error * output_phi_prime(u) * (-x_i) # update output weights

                l1_error = d_i - y_1

                W_1[1] += eta * l1_error * phi_prime(v) * (-x_i) # update input weights
                W_1[0] += eta * l1_error # update input biases

            # Mean Square Error
            new_mse = np.mean((d_i_n - y_2)) ** 2
            mean_square_errors[epoch][i] = new_mse
            if new_mse > mse:
                # we are increasing the MSE, so we need to modify eta
                eta *= 0.9
            mse = new_mse

        epoch += 1

    mean_square_errors = np.resize(mean_square_errors, (epoch, n))
    print("final mse: {}".format(mse))
    print("Epochs: {}".format(epoch))
    print("Errors: {}".format(errors[:epoch]))

    for i in range(epoch):
        if i > 0:
            break
        plt.title("n vs mse, epoch=" + str(i))
        plt.xlabel("n")
        plt.ylabel("mse")
        plt.plot([i for i in range(n)], mean_square_errors[i])
        plt.show()
    
    plt.title("missed classifications per epoch")
    plt.xlabel("epoch")
    plt.ylabel("missed classifications")
    plt.plot([i for i in range(epoch)], [errors[i] for i in range(epoch)])
    plt.show()
    
    # Section 5
    W_0 = np.vstack((W_1, W_2))
    x = np.linspace(0,1,n)
    vals = np.zeros(n)
    for i in range(n):
        # another forward pass now that the network is trained
        v = np.add(W_0[0], W_0[1] * x[i]) # local field of first layer, W_0[0] are biases
        y_1 = phi(v) # output of first layer
        u = (W_0[2] * y_1) + output_bias # local field of second layer
        y = output_phi(np.sum(u)) # output of second layer
        vals[i] = y

    plt.title('f(x, w_0), n=' + str(n))
    plt.xlabel='x'
    plt.ylabel='f(x, w_0)'
    plt.scatter(x_1_n, d_i_n)
    plt.scatter(x, vals)
    plt.show()


if __name__ == '__main__':
    main()

# EOF
