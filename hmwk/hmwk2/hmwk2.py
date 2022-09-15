import numpy as np
import matplotlib.pyplot as plt
from random import random, uniform


def step_function(x):
    """ f(x) = 1 if x >= 0
        f(x) = 0 if x < 0
    """
    # TODO is numpy really necessary?
    return np.array(x >= 0, dtype=int)


def neg_one_to_one():
    return uniform(-1, 1)

def main():

    # begin weights
    w_0 = uniform(-1/4, 1/4)
    w_1 = neg_one_to_one()
    w_2 = neg_one_to_one()

    # x_1 ... x_n
    x_1_n = np.array([neg_one_to_one(), neg_one_to_one()])
    n = 99
    for i in range(n):
        x_i = np.array([neg_one_to_one(), neg_one_to_one()])
        x_1_n = np.vstack((x_1_n, x_i))

    # x_1_n should now be a 100x2 matrix

    # determine s_1 and s_0 where
    # s_0 subset of S where x [x_1, x_2] an eleement of S satisfying [1 x_1 x_2][w_0 w_1 w_2]^T < 0
    # s_1 subset of S where x [x_1, x_2] an eleement of S satisfying [1 x_1 x_2][w_0 w_1 w_2]^T >= 0

    a = np.array([1, x_1_n[0][0], x_1_n[0][1]])
    b = np.array([w_0, w_1, w_2])

    s_0 = np.empty((0,2), float)
    s_1 = np.empty((0,2), float)

    for i in range(100):
        a = np.array([1, x_1_n[i][0], x_1_n[i][1]])
        b = np.array([w_0, w_1, w_2])
        if np.sum(a * b) >= 0:
            s_1 = np.vstack((s_1, x_1_n[i]))
        else:
            s_0 = np.vstack((s_0, x_1_n[i]))

    # print(x_1_n[0][0])
    # print(s_0)

    # s_0 should now be the collection of all x[x_1, x_2] an element of S where [1 x_1 x_2][w_0 w_1 w_2]^T < 0
    # s_1 should now be the collection of all x[x_1, x_2] an element of S where [1 x_1 x_2][w_0 w_1 w_2]^T >= 0


    # Plotting
    fig = plt.figure(figsize=(10,8))

    # datapoints
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.title('Random Classification Data with 2 classes')
    plt.plot(s_0[:, 0], s_0[:, 1], 'r^')
    plt.plot(s_1[:, 0], s_1[:, 1], 'bs')

    # decision boundary
    x1 = np.arange(-1,1,0.01)
    plt.plot(x1, (-w_0-w_1*x1)/w_2, 'k-')

    # render plot
    plt.show()


# Perceptron training algorithm in plain english
# 1. Initialize weights to random values
# 2. For each training example (x, d)
# 3. Compute the output value of the perceptron: y = f(w^T * x)
# 4. Update the weights: w = w + (d - y) * x
# 5. Repeat steps 2-4 until all training examples are classified correctly



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
