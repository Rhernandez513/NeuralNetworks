import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

x = (torch.arange(50)+1.0).to(device)
d = x+(2*(torch.rand(50).to(device))-1)

x = x.unsqueeze(1)
d = d.unsqueeze(1)



network = torch.nn.Linear(1,1).to(device)
optimizer = torch.optim.SGD(network.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=2000, gamma=0.9)
loss_func = torch.nn.MSELoss()


for t in range(1000):
    y = network(x)
    loss = loss_func(y, d)

    if(t%100==0):
        print(t,loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

plt.figure(figsize=(10,4))
plt.plot(x.detach().cpu(), network(x).detach().cpu(), 'g-')
plt.plot(x.detach().cpu(), d.detach().cpu(), 'r.')
plt.show()
