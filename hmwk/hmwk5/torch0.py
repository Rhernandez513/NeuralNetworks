import torch
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

x = (torch.arange(50)+1.0).to(device)
d = x+2*(torch.rand(50).to(device))-1

# my fit y = a*x_axis + b

a = torch.tensor(5.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

eta = 0.001

for t in range(100):
    y = a*x + b
    loss = torch.sum((y-d)*(y-d)/50)

    a.retain_grad()
    b.retain_grad()

    loss.backward()

    a = a - eta*a.grad
    b = b - eta*b.grad


plt.figure(figsize=(10,4))
plt.plot(x.detach().cpu(), y.detach().cpu(), 'g-')
plt.plot(x.detach().cpu(), d.detach().cpu(), 'r.')
plt.show()
