import torch
import numpy as np
import torch.nn as nn

class Network(object):
    def __init__(self, args):
        self.args = args

        self.args['layers'] = [10 for _ in range(args['num_layers'])]
        self._build_network()

    def _build_network(self):
        self.layers = []
        input_shape = self.args['input_shape']
        for N in self.args['layers']:
            self.layers.append(
                self._init_layer(input_shape, (N, N))
            )

            input_shape = (N, N)

    def _init_layer(self, input_shape, layer_shape):
        layer = {
            'weights':{
                'proximal':self._build_weights(layer_shape, input_shape),            # bottom up
                'distal':self._build_weights(layer_shape, layer_shape),              # intra layer
                'apical':self._build_weights(input_shape, layer_shape)               # top down
            },
            'connections':{
                'proximal':self._build_layer_connections(layer_shape, input_shape),
                'distal':self._build_layer_connections(layer_shape, layer_shape),
                'apical':self._build_layer_connections(input_shape, layer_shape)
            }
        }
        return layer

    def _build_weights(self, output_shape, input_shape):
        return torch.randn(output_shape+input_shape)

    def _build_layer_connections(self, output_shape, input_shape):
        connections = np.zeros(output_shape + input_shape)
        for row in range(output_shape[0]):
            for col in range(output_shape[1]):
                connections[row][col] = self._generate_mask(row, col, input_shape[0])
        return connections

    def _generate_mask(self, row, col, n, r=3):
        y,x = np.ogrid[-row:n-row, -col:n-col]
        mask = x*x + y*y <= r*r

        array = np.zeros((n, n))
        array[mask] = 1
        return array

    def forward(self, x):



    def learn(self):
        for l_idx in range(self.num_layers):
            layer = self.W[l_idx]
            active_neurons = self.active_neurons[l_idx]

            for col,act_neuron in zip(layer,active_neurons):
                pass
