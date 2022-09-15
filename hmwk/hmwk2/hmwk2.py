#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import random


def step_function(x):
    """ f(x) = 1 if x >= 0
        f(x) = 0 if x < 0
    """
    # TODO is numpy really necessary?
    return np.array(x >= 0, dtype=int)


def plot_data(s_0, s_1, w_0, w_1, w_2):
    # Plotting training data
    fig = plt.figure(figsize=(10,8))

    # training datapoints
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.title('Random Classification Data with 2 classes')
    plt.plot(s_0[:, 0], s_0[:, 1], 'r^')
    plt.plot(s_1[:, 0], s_1[:, 1], 'bs')

    # decision boundary
    x1 = np.arange(-1,1,0.01)
    # here we plot the decision boundary the perceptron will output
    # after training
    plt.plot(x1, (-w_0 - w_1 * x1)/w_2, 'k-')

    # render plot
    plt.show()

def main():

    # weights
    w_0, w_1, w_2 = random.uniform(-1/4, 1/4), random.uniform(-1, 1), random.uniform(-1, 1)

    # S: x_1 ... x_n
    S = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
    n = 99
    for i in range(n):
        x_i = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
        S = np.vstack((S, x_i))
    # S should now be a 100x2 matrix

    # determine s_1 and s_0 where
    # s_0 subset of S where x=[x_1, x_2] an eleement of S satisfying [1 x_1 x_2][w_0 w_1 w_2]^T < 0
    # s_1 subset of S where x=[x_1, x_2] an eleement of S satisfying [1 x_1 x_2][w_0 w_1 w_2]^T >= 0

    input_vector = np.array([1, S[0][0], S[0][1]])
    weight_vector = np.array([w_0, w_1, w_2])

    s_0 = np.empty((0,2), float)
    s_1 = np.empty((0,2), float)

    for i in range(100):
        input_vector = np.array([1, S[i][0], S[i][1]])
        weight_vector = np.array([w_0, w_1, w_2])
        if np.sum(input_vector * weight_vector) >= 0:
            s_1 = np.vstack((s_1, S[i]))
        else:
            s_0 = np.vstack((s_0, S[i]))

    # s_0 should now be the collection of all x[x_1, x_2] an element of S where [1 x_1 x_2][w_0 w_1 w_2]^T < 0
    # s_1 should now be the collection of all x[x_1, x_2] an element of S where [1 x_1 x_2][w_0 w_1 w_2]^T >= 0

    # section g
    plot_data(s_0, s_1, w_0, w_1, w_2)

    # we will use s_0 and s_1 to train the perceptron

    eta = 1
    misclassifications = 0
    epochNumber = 0


# Perceptron training algorithm in plain english
# 1. Initialize weights to random values
# 2. For each training example (x, d)
# 3. Compute the output value of the perceptron: y = f(w^T * x)
# 4. Update the weights: w = w + (d - y) * x
# 5. Repeat steps 2-4 until all training examples are classified correctly

    w_0, w_1, w_2 = random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)

    weight_vector = np.array([w_0, w_1, w_2])

    # pseudo do-while loop
    (weight_vector, misclassifications) = train_perceptron(S, s_1, weight_vector, eta)
    print("Epoch: ", epochNumber, end=" ")
    print("Misclassifications: ", misclassifications)
    while misclassifications > 0:
        epochNumber += 1
        (weight_vector, misclassifications) = train_perceptron(S, s_1, weight_vector, eta)
        print("Epoch: ", epochNumber, end=" ")
        print("Misclassifications: ", misclassifications)
    print("Converged after ", epochNumber, " epochs")
    print("Final weights: ", weight_vector)



    # TODO Render the decision boundary of the perceptron after training

def train_perceptron(S, s_1, weight_vector,eta):
    misclassifications = 0
    for i in range(len(S)):

        input_vector = np.array([1, S[i][0], S[i][1]])

        # y = u(Omega T x_i)
        y = np.sum(input_vector * weight_vector)
        # Perceptron output
        y = 1 if y >= 0 else 0

        # d_i is the actual classification of the data point 
        # we must compare the classification computer by the Perceptron
        # to the actual classification
        d_i = 1 if np.array([S[i][0], S[i][1]]) in s_1 else 0

        # we'll write the "longer" algorithm here first, then upgrade to the shorthand after
        if y == 1 and d_i == 0:
            weight_vector[0] -= eta * input_vector[0]
            weight_vector[1] -= eta * input_vector[1]
            weight_vector[2] -= eta * input_vector[2]
            misclassifications += 1
        elif y == 0 and d_i == 1:
            weight_vector[0] += eta * input_vector[0]
            weight_vector[1] += eta * input_vector[1]
            weight_vector[2] += eta * input_vector[2]
            misclassifications += 1
    return (weight_vector, misclassifications)

# We can use the Perceptron Training Algorithm (PTA) that finds the weight vector for us.  Of course, it is a special case of supervised learning
#     - Supposing n training samples x1 , … , xn ∈ ℝ1+d are given.  Again assume the first component of these vectors are assumed to be equal to 1 to provide for the bias.
#     - Let d1 , … , dn ∈ { 0,1 } represent the desired output for each of the input vectors
#     - We have, C0 = {xi : di  = 0} and C1 = {xi  : di = 1}
#     - Suppose C0 and C1 are linearly separable.
#     - Consider any learning parameter 𝜂 > 0. Then, the following algorithm finds a weight vector 𝛺 ∈ ℝ1+d that can separate the two classes:
#     1) Init 𝛺 arbitrarily (e.g. randomly)
#     2) epochNumber ← 0
#     3) While 𝛺 cannot correctly classify all input patterns, i.e. while u(𝛺Txi) ≠ di for some i ∈ {1 ,… ,n}
#         a) epochNumber ← epochNumber + 1 
#         b) For i = 1 to n
#             i) Calculate y = u(𝛺Txi) (the output for the ith training sample with the current weights
#             ii) If y=1 but di=0
#                 A. Update the weights 𝛺 ← 𝛺 - 𝜂xi 
#             iii) If y=0 but di = 1
#                 A. Update the weights as 𝛺 ← 𝛺 + 𝜂xi
# Of course epochNumber is only for tracking
#  -  We can write the Algo a bit more simply:
    # 1. Init 𝛺 randomly
    # 2. While 𝛺 cannot correctly classify all input patterns, i.e. while u(𝛺Txi) ≠ di for some i ∈ {1, … , n}
    # 	a) For i = 1 to n 
    # 		i) 𝛺 ← 𝛺 + 𝜂xi (di - u(𝛺Txi)) 


class Perception:
    def __init__(self):
        pass
    def predict(self, x):
        pass
    
def perceptron_training_algorithm(perceptron, data):
    pass

def organize_data():
    pass


if __name__ == "__main__":
    main()

# EOF
