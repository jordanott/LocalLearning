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

parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs used for training')
parser.add_argument('--env', type=str, default='MNIST', choices=['MNIST', 'LMNIST', 'GYM'], help='Type of task')
parser.add_argument('--vis_weights', default=False, action='store_true', help='Store visualizations of weights')

args = vars(parser.parse_args())

env = EnvManager(args)
monitor = AgentMonitor(args)
net = Network(args)

if __name__ == '__main__':
    state = env.reset()

    for i in range(2000):
        action = net.act(state)

        state = env.step(action)

    for i in range(10):
        action = net.act(state)

        state = env.step(action)

        net.vis_layers()

    if args['vis_weights']: net.layers[0].vis_weights(train='post')

    monitor.update(net)
