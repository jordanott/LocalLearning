import matplotlib.pyplot as plt

class AgentMonitor(object):
    def __init__(self, args):
        # self.layers = {l:{c:[0]*layer_info[l] for c in range(num_classes)} for l in range(len(layer_info))}
        self.args = args
    def add(self, layer):
        pass

    def plot(self):
        for i in self.layers:
            plt.bar()
