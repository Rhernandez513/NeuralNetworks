# Some raw output from GitHub CoPilot 

# Perceptron Training Algorithm
# Path: main.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# Data
x_train = torch.FloatTensor([[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]])
y_train = torch.FloatTensor([[0], [0], [0], [1], [1], [1]])

# Model
W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# Optimizer
optimizer = optim.SGD([W, b], lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    
        # H(x) 계산
        hypothesis = torch.sigmoid(x_train.matmul(W) + b)
    
        # cost 계산
        cost = -(y_train * torch.log(hypothesis) +
                (1 - y_train) * torch.log(1 - hypothesis)).mean()
    
        # cost로 H(x) 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
    
        if epoch % 100 == 0:
            print('Epoch {:4d}/{} Cost: {:.6f}'.format(
                epoch, nb_epochs, cost.item()
            ))

# Perceptron Training Algorithm
