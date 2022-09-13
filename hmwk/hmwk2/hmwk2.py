import numpy as np
import matplotlib.pyplot as plt



# from lecture notes

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

def main():
    print("Hello World")

if __name__ == "__main__":
    main()

# EOF
