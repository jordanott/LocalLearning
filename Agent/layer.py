import torch
import numpy as np
from torch import nn

class Layer:
    def __init__(self, input_shape, layer_shape, top_down_shape):
        self.weights = {
            'proximal':self._build_weights(layer_shape, input_shape),            # bottom up
            'distal':self._build_weights(layer_shape, layer_shape),              # intra layer
            'apical':self._build_weights(layer_shape, top_down_shape)            # top down
        }
        self.connections = {
            'proximal':self._build_layer_connections(layer_shape, input_shape),
            'distal':self._build_layer_connections(layer_shape, layer_shape),
            'apical':self._build_layer_connections(layer_shape, top_down_shape)
        }

        self.state = torch.zeros(layer_shape, dtype=torch.float64)
        self.pool = nn.MaxPool2d((2,2))

    def _build_weights(self, output_shape, input_shape):
        # no weights for connections; only used for top layer
        if input_shape is None: return None

        return torch.randn(output_shape+input_shape, dtype=torch.float64)

    def _build_layer_connections(self, output_shape, input_shape):
        # no weights for connections; only used for top layer
        if input_shape is None: return None

        connections = np.zeros(output_shape + input_shape)
        for row in range(output_shape[0]):
            for col in range(output_shape[1]):
                connections[row][col] = self._generate_mask(row, col, input_shape[0])
        return torch.from_numpy(connections)

    def _generate_mask(self, row, col, n, r=3):
        y,x = np.ogrid[-row:n-row, -col:n-col]
        mask = x*x + y*y <= r*r

        array = np.zeros((n, n))
        array[mask] = 1
        return array

    def _compute(self, x, dendrite):
        connections = x * self.connections[dendrite]
        activities = torch.einsum('rchw,rchw->rc', (connections, self.weights[dendrite]))
        return activities

    def _update_state(self, x):
        self.state += x

    def _fire(self):
        max_values = self.pool(self.state)
        active_neurons = max_values == self.state
        return active_neurons

    def layer_compute(self, x, layer_above):
        top_down = self._compute(layer_above, 'apical')

        #
        self._update_state(top_down)

        intra_layer = self._compute(self.state, 'distal')

        self._update_state(intra_layer)

        # compute bottom up activity
        bottom_up = self._compute(x, 'proximal')

        self._update_state(bottom_up)

        return self._fire()
