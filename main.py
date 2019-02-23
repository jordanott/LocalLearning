import torch
import argparse
import numpy as np
import torch.nn as nn

from Agent.model import Network
from Monitor.monitor import AgentMonitor
from torchvision import datasets, transforms
from Env.environment_manager import EnvManager
# set random seeds
np.random.seed(0); torch.manual_seed(0)

parser = argparse.ArgumentParser()

parser.add_argument('--num_layers', type=int, default=5, help='Number of layers')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs used for training')
parser.add_argument('--env', type=str, default='MNIST', choices=['MNIST', 'LMNIST', 'GYM'], help='Type of task')

args = vars(parser.parse_args())

env = EnvManager(args)
monitor = AgentMonitor(args)
net = Network(args)

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
