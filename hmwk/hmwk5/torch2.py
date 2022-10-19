import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

x = (torch.arange(50)+1.0).to(device)
d = x+10*(2*(torch.rand(50).to(device))-1)
#d = d[torch.randperm(50)]

x = x.unsqueeze(1)
d = d.unsqueeze(1)

class MyNetwork(torch.nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.fc1 = nn.Linear(1, 5000)
        self.fc3 = nn.Linear(5000, 500)
        self.fc2 = nn.Linear(500, 1)

    def forward(self, x):
        x = self.fc1((x-25.0)/5.0)
        x = F.tanh(x)
        x = self.fc3(x)
        x = F.tanh(x)
        x = 50*self.fc2(x)
        return x


network = MyNetwork().to(device)
optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=2000, gamma=0.9)
loss_func = torch.nn.MSELoss()


for t in range(100000):
    y = network(x)
    loss = loss_func(y, d)

    if(t%100==0):
        print(t,loss)
    if(loss < 1e-4):
        break

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

plt.figure(figsize=(10,4))
fine_x = torch.arange(1, 50.0, 0.01).to(device)
fine_x = fine_x.unsqueeze(1)
plt.plot(fine_x.detach().cpu(), network(fine_x).detach().cpu(), 'g-')
plt.plot(x.detach().cpu(), d.detach().cpu(), 'r.')
plt.show()
