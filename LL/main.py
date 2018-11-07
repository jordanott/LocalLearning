import torch
import numpy as np
import torch.nn as nn

from network import Network
from torchvision import datasets, transforms

num_classes = 10; layer_info = [20, 10]; input_shape = 784

net = Network(num_classes, layer_info, input_shape)

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("../data", train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           #transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=1, shuffle=True)

for batch_idx, (data, target) in enumerate(train_loader):
    x = data.view(-1, 784)

    print net.forward(x)

    break
