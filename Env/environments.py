import gym
import torch
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
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

        self.train_idx = 0
        self.test_idx = 0

        self.hash = {}
        for k in np.unique(self.y_train):
            self.hash[k] = torch.zeros(100, dtype=torch.uint8)

    def step(self, action, kwargs):
        RECORD = kwargs.get('RECORD', False)
        # yield next image; not interactible
        if RECORD:
            label = self.y_train[self.train_idx]
            self.hash[label] = action | self.hash[label]

        self.train_idx += 1
        return self._get_train_state()

    def eval(self, action, label_similarity={}):
        label = self.y_test[self.test_idx]
        # iterate through the categories to find the most similar one
        for k in self.hash:
            label_similarity[k] = torch.sum(action & self.hash[label])

        # the prediction is the category with most overlap
        prediction = max(label_similarity, key=label_similarity.get)

        # evaluate prediction
        self.correct_predictions += prediction == label

        self.test_idx += 1
        return self._get_test_state()

    def reset(self, kwargs):
        TRAIN = kwargs.get('TRAIN', False)
        if TRAIN:
            self.train_idx = 0
            return self._get_train_state()
        else:
            self.test_idx = 0; self.correct_predictions = 0
            return self._get_test_state()

    def _get_train_state(self):
        return (self.x_train[self.train_idx] > 1).astype(np.float16)

    def _get_test_state(self):
        return (self.x_test[self.test_idx] > 1).astype(np.float16)

class SMNIST(Environment):
    def __init__(self, args):
        super(MNIST, self).__init__(args)
        self.args['input_shape'] = (28,28)
        (self.data, y_train), (x_test, y_test) = mnist.load_data()
        self.data_index = 0
        self.num_samples = 10

    def step(self, action):
        # yield next image; not interactible
        self.data_index += 1
        return self._get_state()

    def reset(self):
        self.data_index = 0
        return self._get_state()

    def _get_state(self):
        return (self.data[self.data_index % self.num_samples] > 1).astype(np.float16)

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
