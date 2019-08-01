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
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs used for training')
parser.add_argument('--env', type=str, default='MNIST', choices=['MNIST', 'SMNIST', 'LMNIST', 'GYM'], help='Type of task')
parser.add_argument('--vis_weights', default=False, action='store_true', help='Store visualizations of weights')

args = vars(parser.parse_args())

env = EnvManager(args)
monitor = AgentMonitor(args)
net = Network(args)

if __name__ == '__main__':
    state = env.reset()

    for i in range(args['epochs']):
        action = net.act(state)
        state = env.step(action)

    for i in range(100):
        action = net.act(state)
        state = env.step(action, RECORD=True)

    for i in range(50):
        action = net.act(state)
        state = env.eval(action)
        # net.vis_layers()

    print 'Accuracy:', env.env.correct_predictions / float(env.env.test_idx)

    if args['vis_weights']: net.layers[0].vis_weights(train='post')

    monitor.update(net)
