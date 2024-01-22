import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __int__(self):
        super().__int__()
        self.fc = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))

    def forward(self, x):
        return self.fc(x)

net1 = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
net2 = MLP()

print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name())
print(torch.cuda.current_device())
x = torch.tensor([[1, 2, 3], [4, 5, 6]]).cuda()
print(x.device)

X = torch.rand(2, 4)
print(X)
print(net1(X))
# print(net2.forward(X))

print(net1[2].state_dict())
print(net1[2].bias)
print(net1[2].bias.data)

