from network import Network
from layers.one_d import one_d
from layers.two_d import two_d
from data import Data
import matplotlib.pyplot as plt
import numpy as np

VISUALIZE = 'last'
INPUT_SHAPE = (28,28)
I = Data(INPUT_SHAPE,mnist=True)

net = Network()
'''
net.add(one_d(), input_shape=INPUT_SHAPE)
net.add(one_d())
net.add(one_d())
net.add(one_d(receptive_field=100))
'''
net.add(two_d(receptive_field=5), input_shape=INPUT_SHAPE)
net.add(two_d(receptive_field=7))
#net.add(two_d(receptive_field=10))
net.summary()

# train
for _ in range(1):
    for i in range(len(I.train)):
        if (i+1) % 1000 == 0:
            net.update_weights()

            net.test_representations(I.test,I.test_label,10)
            net.representations = {}

            net.forward(I.train[np.random.randint(0,len(I.train))],visualize=VISUALIZE)            
        net.forward(I.train[i],visualize=False)

print 'Training finished'

#net.hash_representations(I.train)

net.test_representations(I.test,I.test_label,10,plot=True)
net.plot_weight_history()
