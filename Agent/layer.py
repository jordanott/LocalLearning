import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn

class Layer:
    def __init__(self, args, ID, input_shape, layer_shape, top_down_shape,
        activity_decay=0.9, threshold=.2, learning_rate=1e-3, reset_potential=0):
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

        self.ID = ID
        self.threshold = threshold
        self.layer_shape = layer_shape
        self.learning_rate = learning_rate
        self.unpool = nn.MaxUnpool2d((2,2))
        self.activity_decay = activity_decay
        self.reset_potential = reset_potential
        self.pool = nn.MaxPool2d((2,2),return_indices=True)
        self.state = torch.zeros(layer_shape, dtype=torch.float64)
        self.active_neurons = torch.zeros(layer_shape, dtype=torch.float64)

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
        if args['vis_weights']:  self.vis_weights()

    def build_directory(self, path, current_path=''):
        print path
        # iterate through folders in specifide path
        for folder in path.split('/'):
            current_path += folder +'/'
            # if it doesn't exist build that director
            if not os.path.exists(current_path):
                os.mkdir(current_path)

    def vis_weights(self, train='pretrain'):
        def vis(dendrite_type):
            path = 'Results/Weights/{}/{}/{}/'.format(train, self.ID, dendrite_type)
            self.build_directory(path)

            for row in range(self.weights[dendrite_type].shape[0]):
                for col in range(self.weights[dendrite_type].shape[1]):
                    plt.clf()

                    plt.subplot(1,3,1)
                    plt.title(dendrite_type + ' weights')
                    plt.imshow(self.weights[dendrite_type][row,col])

                    plt.subplot(1,3,2)
                    plt.title(dendrite_type + ' connections')
                    plt.imshow(self.connections[dendrite_type][row,col])

                    plt.subplot(1,3,3)
                    plt.title(dendrite_type + ' actual')
                    plt.imshow(self.weights[dendrite_type][row,col] * self.connections[dendrite_type][row,col])

                    plt.colorbar()
                    plt.savefig(path + '%03d_%d.png' % (row, col))

        vis('basal'); vis('intra')
        if self.weights['apical'] is not None: vis('apical')

    def _build_weights(self, output_shape, input_shape):
        # no weights for connections; only used for top layer
        if input_shape is None: return None

        return 0.1 * torch.randn(output_shape+input_shape, dtype=torch.float64)

    def _build_layer_connections(self, output_shape, input_shape):
        # no weights for connections; only used for top layer
        if input_shape is None: return None

        connections = np.zeros(output_shape + input_shape)
        scale_row = input_shape[0] / float(output_shape[0])
        scale_col = input_shape[1] / float(output_shape[1])
        for row in range(output_shape[0]):
            for col in range(output_shape[1]):
                connections[row][col] = self._generate_mask(int(scale_row*row), int(scale_col*col), input_shape[0])
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

        # reset neurons after spike
        self.state[active_neurons] =  self.reset_potential

        return active_neurons

    def layer_compute(self, x, layer_above):
        self.x = x; self.layer_above = layer_above
        self.previous_state = self.state.clone(); self.previous_active_neurons = self.active_neurons.clone()
        # tow down feedback
        if layer_above is not None:
            self.top_down = self._compute(layer_above, 'apical')
            self._update_state(self.top_down)

        # intra layer
        self.intra_layer = self._compute(self.active_neurons, 'intra')
        self._update_state(self.intra_layer)

        # compute bottom up activity
        self.bottom_up = self._compute(x, 'basal')
        self._update_state(self.bottom_up)

        self.active_neurons = self._fire()

        if layer_above is not None:
            self._learn(layer_above, self.active_neurons, 'apical')
        self._learn(self.active_neurons, self.active_neurons, 'intra')
        self._learn(x, self.active_neurons, 'basal')

        return self.active_neurons

    def _learn(self, input, output, dendrite_type):
        self._hebbian(input, output, dendrite_type)
        self._anti_hebbian(input, output, dendrite_type)

        norm = torch.norm(self.weights[dendrite_type].view(1,-1), p=2).sum().data.numpy()
        sparsity = output.nonzero().size(0) / float(output.nelement())

        self.layer_info['norms'][dendrite_type].append(norm)
        self.layer_info['sparsity'][dendrite_type].append(sparsity)

        self.weights[dendrite_type] = torch.clamp(self.weights[dendrite_type], -1, 1)

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

    def vis_activity(self):
        fig = plt.figure(figsize=(10,10))
        if self.layer_above is not None:
            plt.subplot(3,3,1); plt.title('Input'); plt.imshow(self.layer_above); plt.colorbar()
            plt.subplot(3,3,2); plt.title('Apical Output'); plt.imshow(self.top_down); plt.colorbar()
        plt.subplot(3,3,3); plt.title('Active Neurons'); plt.imshow(self.active_neurons); plt.colorbar()

        plt.subplot(3,3,4); plt.title('Input'); plt.imshow(self.active_neurons); plt.colorbar()
        plt.subplot(3,3,5); plt.title('Intra Output'); plt.imshow(self.intra_layer); plt.colorbar()
        plt.subplot(3,3,6); plt.title('Previously Activie Neurons'); plt.imshow(self.previous_active_neurons); plt.colorbar()

        plt.subplot(3,3,7); plt.title('Input'); plt.imshow(self.x); plt.colorbar()
        plt.subplot(3,3,8); plt.title('Basal Output'); plt.imshow(self.bottom_up); plt.colorbar()
        plt.subplot(3,3,9); plt.title('Previous State'); plt.imshow(self.previous_state); plt.colorbar()

        fig.suptitle('Layer %d' % self.ID)
        plt.show()
