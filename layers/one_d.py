import numpy as np
import matplotlib.pyplot as plt

class one_d:
    def __init__(self,receptive_field=200,activity_rate=.1,locally_connected=True):
        self.lr = 1e-2
        self.W_history = []
        self.activity_rate = activity_rate
        self.receptive_field = receptive_field
        self.locally_connected = locally_connected

    def _get_layer_shape(self):
        # default stride one
        stride = 1
        shape = ((self.input_shape[1] - self.receptive_field)/stride) + 1
        return shape

    def init_layer(self,input_shape,layer_num):
        self.input_shape = input_shape
        self.num_synapses = input_shape[1]
        self.num_neurons = self._get_layer_shape()
        self.layer_shape = (1,self.num_neurons)
        self.active_neurons = int(self.activity_rate * self.num_neurons)
        self.layer_num = layer_num

        if self.locally_connected:
            self.W = self._init_weights()
        else:
            self.W = np.random.uniform(0,1,size=(self.num_neurons,self.num_synapses))

    def _init_weights(self):
        window = self.num_synapses / self.num_neurons
        overlap = 100
        w = np.random.uniform(0,1,size=(self.num_neurons,self.num_synapses))
        m = np.zeros((self.num_neurons,self.num_synapses))
        i = 0
        for n in range(self.num_neurons):
            start = min(i,overlap)
            m[n,i-start:i+1] = 1
            end = min(self.num_synapses-i,overlap) + 1
            m[n,i:i+end] = 1
            i += 1
        w *= m
        return w
        
    def _s(self,x):
        return 1/(1 + np.exp(-x))

    def _ds(self,x):
        return self._s(x)*(1 - self._s(x))

    def forward(self,I):
        V = self._s(self.W)*I
        # activity of each neuron ~ shape: (num_neurons)
        VP = np.sum(V,axis=1) / self.num_synapses
        # sort by activity (indexes): low to high
        A = np.argsort(VP)[-self.active_neurons:]

        _w = self.W[A]
        _v = V[A]
        _w[_v  > 0] += self._ds(_w[_v  > 0])*self.lr

        __w = _w[_v == 0]
        __w[__w != 0] -= self._ds(__w[__w != 0])*self.lr
        _w[_v == 0] = __w
        self.W[A] = _w

        self.W_history.append(np.copy(self.W))
        # set active neurons
        VP[:] = 0
        VP[A] = 1

        return VP

    def forward_(self,I):
        # forward without learning
        V = self._s(self.W)*I
        # activity of each neuron ~ shape: (num_neurons)
        VP = np.sum(V,axis=1) / self.num_synapses
        # sort by activity (indexes): low to high
        A = np.argsort(VP)[-self.active_neurons:]

        # set active neurons
        VP[:] = 0
        VP[A] = 1

        return VP

    def plot_weight_history(self):
        import os
        # create directory for layer weight plots
        if not os.path.exists(str(self.layer_num)):
            os.mkdir(str(self.layer_num))

        W_history = np.array(self.W_history)
        print W_history.shape
        # iterate through neurons in layer_num
        for n in range(self.num_neurons):
            # plot all incoming weights for that neuron on same plot
            synapses = np.arange(self.num_synapses)
            np.random.shuffle(synapses)
            for i in synapses[:self.num_synapses/3]:
                # indexed by: sample,nth neuron,ith weight
                plt.plot(W_history[:,n,i])

            plt.savefig(str(self.layer_num)+'/neuron_{}.png'.format(n))
            plt.clf()
