import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# EOS char => #

# define a recurrent neural network 
def RNN(x, weights, biases):
	# use the RNN to predict the next letter in the sequence

	# ok so we are going to have to take in the input names, reshape to 1xN and then feed into the RNN

	# we will have to design a simple RNN that takes in a sequence of letters and predicts the next letter in the sequence

	pass

# ok so how are we going to measuire the loss of the RNN? 
# copilot suggestion => potentially use the pytorch cross entropy loss function

class BasicLSTM(nn.Module):
	def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
		# github copilot suggestion
		super(BasicLSTM, self).__init__()
		self.num_classes = num_classes
		self.num_layers = num_layers
		self.input_size = input_size
		self.hidden_size = hidden_size
		# I figure we will need this one, should be 27 for alpahebet plus EOS
		# self.seq_length = 27
		self.seq_length = seq_length
		self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
		# self.fc_1 = nn.Linear(hidden_size, 64)
		# self.fc_2 = nn.Linear(64, num_classes)
		# starting with a smaller number of nodes to keep training time down
		self.fc_1 = nn.Linear(hidden_size, 16)
		self.fc_2 = nn.Linear(16, num_classes)

		# this one wasn't suggested by copilot, not needed? a ReLu can't hurt though..
		self.relu = nn.ReLU()

	def fwd(self, x):
		# github copilot suggestion
		# Set initial states
		h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
		c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

		# Forward propagate LSTM
		# out, _ = self.lstm(x, (h0, c0)) #lstm 
		# do we not need cn? 
		out, (hn, cn) = self.lstm(x, (h0, c0)) #lstm 
		out = self.relu(hn)
		out = self.fc_1(out)
		out = self.relu(out)
		out = self.fc_2(out)
		return out

	pass 


def get_names_from_file():
	with open("names.txt") as f:
		lines = f.readlines()
	for idx, val in enumerate(lines):
		lines[idx] = val.lower()
	return lines

# todo generate this programatically so we don't have a huge sparse matrix in source
alphas = {
	'a'	: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
	'b'	: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
	'c'	: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
	'd'	: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
	'e'	: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
	'f'	: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
	'g'	: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
	'h'	: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
	'i'	: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
	'j'	: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
	'k'	: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
	'l'	: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
	'm'	: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
	'n'	: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
	'o'	: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
	'p'	: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
	'q'	: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
	'r'	: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
	's'	: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
	't'	: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 
	'u'	: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 
	'v'	: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 
	'w'	: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 
	'x'	: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
	'y'	: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 
	'z'	: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 
	'#'	: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
}

def name_to_vector_repr(name: str) -> np.array:
	eos_char = np.array(alphas['#'])

	vector_repr = []
	for char in name:
		if char == '\n':
			vector_repr.append(eos_char)
			continue
		vector_repr.append(np.array(alphas[char]))

	if len(vector_repr) < 11:
		for i in range(11 - len(vector_repr)):
			vector_repr.append(eos_char)

	return np.array(vector_repr)

def main():
	# vector representation of the letter a

	names = get_names_from_file()

	names_vector = np.array([name_to_vector_repr(name) for name in names])

	# x = names_vector
	# x_train = names[0:0.8*length]
	# x_test = names[0.8*length:length]
	# def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
	ltsm = BasicLSTM(27, 27, 27, 11, 11)	



main()

# EOF
