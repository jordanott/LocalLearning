import torch
import numpy as np
from torch import nn

class Layer:
    def __init__(self, input_shape, layer_shape, top_down_shape,
        activity_decay=0.9, threshold=1., learning_rate=1e-3):
        self.weights = {
            'basal':self._build_weights(layer_shape, input_shape),            # bottom up
            'intra':self._build_weights(layer_shape, layer_shape),              # intra layer
            'apical':self._build_weights(layer_shape, top_down_shape)            # top down
        }
        self.connections = {
            'basal':self._build_layer_connections(layer_shape, input_shape),
            'intra':self._build_layer_connections(layer_shape, layer_shape),
            'apical':self._build_layer_connections(layer_shape, top_down_shape)
        }

        self.layer_shape = layer_shape
        self.activity_decay = activity_decay
        self.state = torch.zeros(layer_shape, dtype=torch.float64)
        self.pool = nn.MaxPool2d((2,2),return_indices=True)
        self.unpool = nn.MaxUnpool2d((2,2))
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.layer_info = {
            'norms':{
                'basal': [],
                'intra': [],
                'apical': []
            },
            'sparsity':{
                'basal': [],
                'intra': [],
                'apical': []
            }
        }

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

    def _compute(self, x, dendrite_type):
        x = x.double()
        connections = x * self.connections[dendrite_type]
        activities = torch.einsum('rchw,rchw->rc', (connections, self.weights[dendrite_type]))
        return activities

    def _update_state(self, x):
        self.state = self.activity_decay*self.state + x

    def _fire(self):
        # extend dimensions
        state = self.state[None,None]
        # get the most active neurons
        max_values, indices = self.pool(state)
        # print max_values, indices
        max_values = self.unpool(max_values, indices)
        # binary coded input for next layer
        active_neurons = ((max_values == state) & ( max_values > self.threshold)).squeeze()

        return active_neurons

    def layer_compute(self, x, layer_above):
        # tow down feedback
        top_down = self._compute(layer_above, 'apical')
        self._update_state(top_down)
        self._learn(layer_above, top_down, 'apical')

        # intra layer
        intra_layer = self._compute(self.state, 'intra')
        self._update_state(intra_layer)
        self._learn(self.state, intra_layer, 'intra')

        # compute bottom up activity
        bottom_up = self._compute(x, 'basal')
        self._update_state(bottom_up)
        self._learn(x, bottom_up, 'basal')

        return self._fire()

    def _learn(self, input, output, dendrite_type):
        self._hebbian(input, output, dendrite_type)
        self._anti_hebbian(input, output, dendrite_type)

        norm = torch.norm(self.weights[dendrite_type].view(1,-1), p=2).sum().data.numpy()
        sparsity = output.nonzero().size(0) / float(output.nelement())
        print output
        self.layer_info['norms'][dendrite_type].append(norm)
        self.layer_info['sparsity'][dendrite_type].append(sparsity)

    def _hebbian(self, input, output, dendrite_type):
        input = input.double(); output = output.double()
        pre_post_prod = torch.einsum('rc,hw->rchw', (output, input))
        pre_post_prod = pre_post_prod * self.connections[dendrite_type]

        self.weights[dendrite_type] += self.learning_rate * pre_post_prod

    def _anti_hebbian(self, input, output, dendrite_type):
        input = input.double(); output = output.double()

        input_clone = input.clone()
        input_clone[input_clone == 0] = -1; input_clone[input_clone == 1] = 0
        pre = torch.einsum('rc,hw->rchw', (output, input_clone))
        pre = pre * self.connections[dendrite_type]

        output_clone = output.clone()
        output_clone[output_clone == 0] = -1; output_clone[output_clone == 1] = 0
        post = torch.einsum('rc,hw->rchw', (output_clone, input))
        post = post * self.connections[dendrite_type]

        self.weights[dendrite_type] += self.learning_rate * (pre + post)
