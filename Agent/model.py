import os
import torch
import numpy as np
import torch.nn as nn

from layer import Layer

class Network(object):
    def __init__(self, args):
        self.args = args

        self.args['layers'] = [args['neurons'] for _ in range(args['num_layers'])]
        self._build_network()

    def _build_network(self, i=-1):
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
                Layer(self.args, i+1, input_shape, (N, N), (N_1, N_1))
            )
            # output shape of current layer; serves as input to layer above
            input_shape = (N, N)

        N = self.args['layers'][-1]
        # build top layer
        self.layers.append(
            Layer(self.args, i+2, input_shape, (N, N), None) # last layer doesn't get top down input
        )

    def act(self, x, LEARN=True):
        x = torch.from_numpy(x).type(torch.float64)
        for i in range(len(self.layers)):
            layer = self.layers[i]
            top_down = self.layers[i+1].active_neurons if i+1 != len(self.layers) else None

            x = layer.layer_compute(x, top_down, LEARN)

            if self.args['gif_weights']: layer.vis_weights()

        return x.view(-1)

    def gif(self):
        for i in range(len(self.layers)):
            command = 'convert -delay 10 -loop 0 {dir}*.png {dir}layer.gif'.format(
                dir='Results/Weights/%d/' % i
            )
            os.system(command)

    def vis_layers(self):
        for layer in self.layers:
            layer.vis_activity()
