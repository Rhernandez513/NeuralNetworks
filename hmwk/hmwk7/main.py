import numpy as np
import matplotlib.pyplot as plt
import torch


# EOS char => #

# a = [1, 0, 0, 

def get_a():
	vec = np.zeros(27)	
	vec[0] = 1
	return vec


def main():
	a_as_vec = get_a()
	print("len: " + str(len(a_as_vec)))
	print(a_as_vec)


if __name__ == "__main__":
	main()

# EOF

