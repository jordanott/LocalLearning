import sys
import torch
import pprint
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
parser.add_argument("--notsherpa", default=False, action='store_true')
parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs used for training')
parser.add_argument('--neurons', type=int, default=10, help='Total number of neurons is this squared')
parser.add_argument('--vis_weights', default=False, action='store_true', help='Store visualizations of weights')
parser.add_argument('--gif_weights', default=False, action='store_true', help='Store gif visualizations of weights')
parser.add_argument('--env', type=str, default='MNIST', choices=['MNIST', 'SMNIST', 'LMNIST', 'GYM'], help='Type of task')
parser.add_argument('--record', type=str, default='AND', choices=['AND', 'OR'])
args = vars(parser.parse_args())

###########################
import sherpa
client = sherpa.Client(test_mode=args['notsherpa'])
trial = client.get_trial()
###########################

if args['notsherpa']:
    args = trial.parameters
else:
    pass

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(trial.parameters)

env = EnvManager(args)
monitor = AgentMonitor(args)
net = Network(args)

for epoch in range(args['epochs']):
    state = env.reset()
    for i in range(1000):
        action = net.act(state)
        state = env.step(action)

    if args['gif_weights']: net.gif()

    state = env.reset(TRAIN=False)
    for i in range(500):
        action = net.act(state)
        state = env.step(action, RECORD=True)

    for i in range(50):
        action = net.act(state, LEARN=False)
        state = env.eval(action)

    acc = env.env.correct_predictions / float(env.env.test_idx)
    print epoch, acc
    sys.stdout.flush()

    client.send_metrics(trial, epoch, acc, context={})

    if epoch > 1 and acc <= 0.1: break

if args['vis_weights']: net.layers[0].vis_weights(train='post')

# monitor.update(net)
