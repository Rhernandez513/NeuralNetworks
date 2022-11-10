import numpy as np
import matplotlib.pyplot as plt
import torch

# EOS char => #


def get_names():
	with open("names.txt") as f:
		lines = f.readlines()
	for idx, val in enumerate(lines):
		lines[idx] = val.lower()
	return lines

def main():
	# vector representation of the letter a
	vec = np.zeros(27)	
	vec[0] = 1

	eos_char = np.zeros(27)
	eos_char[len(vec) - 1] = 1

	names = get_names()

	print("a: " + str(len(vec)))
	print("eos: " + str(len(eos_char)))
	print(vec)
	print(eos_char)
	print(names)



if __name__ == "__main__":
	main()

# EOF
