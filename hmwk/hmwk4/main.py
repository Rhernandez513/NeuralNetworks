#!/usr/bin/env python3
from re import L
import numpy as np
import matplotlib.pyplot as plt


def get_reals(lower_bound, upper_bound, n) -> np.ndarray:
    return np.random.uniform(lower_bound, upper_bound, n)


def get_d_vector(x, v, n) -> np.ndarray:
    return np.array(
        [np.sin(20 * x[i]) + (3 * x[i]) + v[i] for i in range(n)]
    )

def output_phi(v):
    return v

def phi(v):
    return np.tanh(v)

def main():
    # Notes: We will use a neural network for curve fitting.
    # section 1
    n = 300
    x_1_n = get_reals(0, 1, n)
    # section 2
    v_1_n = get_reals((-1/10), (1/10), n)
    # section 3
    d_i_n = get_d_vector(x_1_n, v_1_n, n)
    plt.title('x v d, n=' + str(n))
    plt.scatter(x_1_n, d_i_n)
    plt.show()

    # Notes
    # We will use a neural network with 1 input
    # 1 hidden layer with 24 neurons
    # 1 output layer with 1 neuron

    # I think this is the hidden layer weights
    W = np.random.uniform(-1, 1, (24, 1))

    # todo look at this in the debugger
    U = np.random.uniform(-1, 1, (1, 24))

    # Section 4 start on the backpropagation algorithm

if __name__ == '__main__':
    main()

# EOF