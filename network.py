import numpy as np
import matplotlib.pyplot as plt
class Network:
    def __init__(self):
        self.layers = []
        self.output_shape = None
        self.input_shape = None
        self.representations = {}

    def add(self,Layer,input_shape=None):
        if self.layers:
            previous_output_shape = self.layers[-1].layer_shape
            Layer.init_layer(previous_output_shape, self.layers[-1].layer_num + 1)
        else:
            self.input_shape = input_shape
            Layer.init_layer(input_shape,1)
        print self.output_shape
        self.output_shape = Layer.layer_shape
        self.layers.append(Layer)
    '''
    def hash_representations(self,data,label,num_classes):
        np.zeros(num_classes)
        # hash representations representations to a category
        for i in range(len(data)):
            string = ''
            for j in freq[i].flatten():
                if j:
                    string += '1'
                else:
                    string += '0'
            self.representations[string] = i
        print len(self.representations)
    '''
    def test_representations(self,data,test,num_classes,plot=False):
        # test
        for i in range(len(data)):
            output = self.forward_(data[i])
            string = ''
            for j in output.flatten():
                if j:
                    string += '1'
                else:
                    string += '0'
            if string not in self.representations:
                self.representations[string] = np.zeros(num_classes)
            self.representations[string][test[i]] += 1
        print 'Learned ',len(self.representations),' representations...'
        #print self.representations
        if plot:
            import matplotlib.pyplot as plt
            counter = 0
            for key in self.representations:
                plt.bar(np.arange(num_classes),self.representations[key])
                plt.savefig(str(counter)+'.png')
                plt.clf()
                counter += 1

    def forward(self,I,visualize=False):
        for l in self.layers:
            if l.layer_num == 1:
                O = l.forward(I,visualize)
            else:
                O = l.forward(O,visualize)
        if visualize == 'last':
                plt.imshow(O)
                plt.title('Layer: '+str(l.layer_num))
                plt.show()
        return O

    def update_weights(self):
        for l in self.layers:
            l.update_weights()
            
    def forward_(self,I):
        for l in self.layers:
            if l.layer_num == 1:
                O = l.forward_(I)
            else:
                O = l.forward_(O)
        return O

    def plot_weight_history(self):
        for l in self.layers:
            l.plot_weight_history()

    def summary(self):
        for l in self.layers:
            print 'Layer:',l.layer_num
            print '\tNeurons:',l.num_neurons
            print '\tWeights:',l.W.shape
            print '\tOutput Shape:',l.layer_shape
