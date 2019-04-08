import torch
import numpy as np
import torch.nn as nn

from layer import Layer

class Network(object):
    def __init__(self, args):
        self.args = args

        self.args['layers'] = [10 for _ in range(args['num_layers'])]
        self._build_network()

    def _build_network(self):
        self.layers = []
        input_shape = self.args['input_shape']
        # first to penultimate layer
        for i in range(len(self.args['layers'])-1):
            # dimensions of current layer
            N = self.args['layers'][i]
            # dimensions of next layer
            N_1 = self.args['layers'][i+1]

            # build layer
            self.layers.append(
                Layer(input_shape, (N, N), (N_1, N_1))
            )
            # output shape of current layer; serves as input to layer above
            input_shape = (N, N)

        N = self.args['layers'][-1]
        # build top layer
        self.layers.append(
            Layer(input_shape, (N, N), None) # last layer doesn't get top down input
        )

    def act(self, x):
        x = torch.from_numpy(x).type(torch.float64)
        for i in range(len(self.layers)-1):
            layer = self.layers[i]
            top_down = self.layers[i+1].state

            x = layer.layer_compute(x, top_down)
        return x
