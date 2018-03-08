import numpy as np
import matplotlib.pyplot as plt

class two_d:
    def __init__(self,receptive_field=10,activity_rate=.2,locally_connected=True,drop_prob=.05):
        self.activity_rate = activity_rate
        self.receptive_field = receptive_field
        self.locally_connected = locally_connected
        self.lr = 1e-2
        self.W_history = []
        self.drop_prob = drop_prob
        self.weight_threshold = 2

    def _get_layer_shape(self):
        # default stride one
        stride = 1
        shape = ((self.input_shape[0] - self.receptive_field)/stride) + 1
        return (shape,shape)

    def init_layer(self,input_shape,layer_num):
        # shape of the input
        self.input_shape = input_shape
        # layer dimensions (n,n)
        self.layer_shape = self._get_layer_shape()
        # total neurons in the layer
        self.num_neurons = self.layer_shape[0]**2
        self.active_neurons = int(self.activity_rate * self.num_neurons)
        # layer number
        self.layer_num = layer_num
        # initialize weights
        self.W = np.random.uniform(-.5,.5,size=(self.layer_shape + self.input_shape))

        if self.locally_connected:
            m = np.zeros((self.layer_shape + self.input_shape))
            for i in range(self.layer_shape[0]):
                for j in range(self.layer_shape[1]):
                    m[i,j,i:i+self.receptive_field,j:j+self.receptive_field] = 1
            self.W *= m
        self._W = self.W

    def _s(self,x):
        return 1/(1 + np.exp(-x))

    def _ds(self,x):
        return self._s(x)*(1 - self._s(x))

    def update_weights(self):
        self.W = self._W

    def visualize(self,I,O):
        plt.imshow(I)
        plt.title('Layer: '+str(self.layer_num) + ' input')
        plt.show()
        plt.imshow(O)
        plt.title('Layer: '+str(self.layer_num) + ' output')
        plt.show()

    def forward(self,I,visualize):
        # dropout
        drop_mask = np.random.choice([False,True],self.W.shape,p=[1-self.drop_prob,self.drop_prob])
        drop = np.copy(self.W)
        drop[drop_mask] = 0
        V = self._s(drop)*I
        # activity of each neuron ~ shape: (num_neurons)
        VP = np.sum(np.sum(V,axis=3),axis=2)
        threshold = np.sort(VP.flatten())[-self.active_neurons]
        # set active neurons
        VP[VP <= threshold] = 0
        VP[VP >  threshold] = 1
        # visualize input output of layer
        if visualize == True:
            self.visualize(I,VP)
        # get weights prior to learning update
        prior_W = np.copy(self._W)
        # learning
        self._W[np.where(VP  > 0)] += self._ds(self._W[np.where(VP  > 0)])*self.lr
        w_zero = np.where(self.W == 0)
        self._W[np.where(VP == 0)] -= self._ds(self._W[np.where(VP == 0)])*self.lr
        self._W[w_zero] = 0
        # reset randomly dropped weights to cancle learning
        self._W[np.where(drop == 0)] = prior_W[np.where(drop == 0)]
        # clip weights
        # self._W = np.clip(self._W,-self.weight_threshold,self.weight_threshold)
        # store weights 
        self.W_history.append(np.copy(self._W))
        return VP

    def forward_(self,I):
        V = self._s(self.W)*I
        # activity of each neuron ~ shape: (num_neurons)
        VP = np.sum(np.sum(V,axis=3),axis=2)
        threshold = np.sort(VP.flatten())[-self.active_neurons]
        # set active neurons
        VP[VP <= threshold] = 0
        VP[VP >  threshold] = 1

        return VP

    def plot_weight_history(self):
        W_history = np.array(self.W_history)
        print W_history.shape
        plt.plot(W_history[:,0,0,0,0])
        plt.show()
        import os
        # create directory for layer weight plots
        if not os.path.exists(str(self.layer_num)):
            os.mkdir(str(self.layer_num))


        num_synapses = self.W.shape[-1]
        synapses = np.arange(num_synapses)
        # iterate through neurons in layer_num
        for n_i in range(self.layer_shape[0]):
            for n_j in range(self.layer_shape[1]):
                # plot all incoming weights for that neuron on same plot
                np.random.shuffle(synapses)
                for i in synapses[:num_synapses/3]:
                    np.random.shuffle(synapses)
                    for j in synapses[:num_synapses/3]:
                        # indexed by: sample,nth neuron,ith weight
                        plt.plot(W_history[:,n_i,n_j,i,j])

                plt.savefig(str(self.layer_num)+'/neuron_{}_{}.png'.format(n_i,n_j))
                plt.clf()
