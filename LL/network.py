import torch
import numpy as np
import torch.nn as nn

from monitor import Monitor

class Network(object):
    def __init__(self, num_classes, layer_info, input_shape, columns=10):
        self.monitor = m = Monitor(num_classes, layer_info)

        self.num_layers = len(layer_info)
        self.W = []
        for i in range(self.num_layers):
            # number of neurons per layer
            N = layer_info[i]

            # input shape for layer
            if i != 0: input_shape = layer_info[i-1]*columns

            self.W.append([torch.randn((input_shape, N)) for _ in range(columns)])

        self.softmax = nn.Softmax()

    def forward(self, x, class=None):
        # record active_neurons per column per layer
        self.active_neurons = [[] for _ in range(self.num_layers)]

        for l_idx in range(self.num_layers):
            layer = self.W[l_idx]
            layer_output = []
            for col in layer:
                act = torch.matmul(x, col)
                act_prob = self.softmax(act).numpy()[0]

                # number of neurons in layer
                N = act_prob.shape[0]

                # sample active neuron according to prob
                act_neuron = np.random.choice(np.arange(N), p=act_prob)

                # output for col is one hot; where active neuron is on ~ 1
                col_output = torch.zeros((1,N))
                col_output[0][act_neuron] = 1

                # output of each col will be input to next layer
                layer_output.append(col_output)

                # record which neurons were active for learning
                self.active_neurons[l_idx].append(act_neuron)

            # set input for next layer
            x = torch.stack(layer_output).view(1,-1)

        return self.active_neurons


    def learn(self):
        for l_idx in range(self.num_layers):
            layer = self.W[l_idx]
            active_neurons = self.active_neurons[l_idx]

            for col,act_neuron in zip(layer,active_neurons):
