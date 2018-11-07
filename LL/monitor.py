import matplotlib.pyplot as plt

class Monitor(object):
    def __init__(self, num_classes, layer_info):
        self.layers = {l:{c:[0]*layer_info[l] for c in range(num_classes)} for l in range(len(layer_info))}

    def add(self, layer):
        pass

    def plot(self):
        for i in self.layers:
            plt.bar()
