import matplotlib.pyplot as plt

class AgentMonitor(object):
    def __init__(self, args):
        # self.layers = {l:{c:[0]*layer_info[l] for c in range(num_classes)} for l in range(len(layer_info))}
        self.args = args
    def update(self, net):
        for i in range(len(net.layers)):
            layer = net.layers[i]

            fig = plt.figure(figsize=(20,10))
            plt.subplot(6,1,1); plt.title('Norm Apical')
            plt.plot(layer.layer_info['norms']['apical'])

            plt.subplot(6,1,2); plt.title('Norm intra')
            plt.plot(layer.layer_info['norms']['intra'])

            plt.subplot(6,1,3); plt.title('Norm basal')
            plt.plot(layer.layer_info['norms']['basal'])

            plt.subplot(6,1,4); plt.title('sparsity Apical')
            plt.plot(layer.layer_info['sparsity']['apical'])

            plt.subplot(6,1,5); plt.title('sparsity intra')
            plt.plot(layer.layer_info['sparsity']['intra'])

            plt.subplot(6,1,6); plt.title('sparsity basal')
            plt.plot(layer.layer_info['sparsity']['basal'])

            fig.subplots_adjust(hspace=.5)
            plt.show()

    def plot(self):
        for i in self.layers:
            plt.bar()
