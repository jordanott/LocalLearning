import gym
import numpy as np
from keras.datasets import mnist

class Environment(object):
    def __init__(self, args):
        self.args = args
    def step(self):
        pass

class MNIST(Environment):
    def __init__(self, args):
        super(MNIST, self).__init__(args)
        self.args['input_shape'] = (28,28)
        (self.data, y_train), (x_test, y_test) = mnist.load_data()
        self.data_index = 0
        self.num_samples = 10

    def step(self, action):
        # yield next image; not interactible
        self.data_index += 1
        self._get_state()

    def reset(self):
        self.data_index = 0
        return self._get_state()

    def _get_state(self):
        return self.data[self.data_index % self.num_samples]

class LMNIST(Environment):
    def __init__(self, args):
        super(LMNIST, self).__init__(args)
        self.args['input_shape'] = (28,28)

    def step(self, action):
        # move eye location depending on action
        # return glimpse at new location
        pass

class GYM(Environment):
    def __init__(self, args):
        super(GYM, self).__init__(args)
        self.args['input_shape'] = (28,28)

    def step(self, action):
        # pass action to gym enviornment
        return self.env.step(action)
