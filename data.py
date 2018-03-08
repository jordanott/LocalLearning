import numpy as np

class Data:

    def __init__(self,shape,mnist=False):
        self.shape = shape
        if mnist:
            from keras.datasets import mnist
            (x_train,y_train),(x_test,y_test) = mnist.load_data()

            x_train = x_train.reshape((60000,)+shape)
            x_test = x_test.reshape((10000,)+shape)
            # threshold
            x_train[x_train <= 120] = 0
            x_train[x_train > 120] = 1

            x_test[x_test <= 120] = 0
            x_test[x_test > 120] = 1
            
            self.train,self.test,self.test_label = x_train[:3003],x_test[:100],y_test[:100]
        else:
            self.train,self.test,self.test_label = self.generate_1d()

    def generate_1d(self):
        # generate train data
        train = []
        for j in range(0,784,28):
            x = np.zeros(784)
            x[j:j+28] = 1
            train.append(x)

        # generate test data
        test,test_label = [],[]
        for j in range(0,784,28):
            x = np.zeros(784)
            if j+29 > 784:
                x[j+1:j+28] = 1
                x[np.random.randint(j,j+28)] = 0
                x[np.random.randint(j,j+28)] = 0
                #x[np.random.randint(j,j+28)] = 0
            else:
                x[j+1:j+29] = 1
                x[np.random.randint(j,j+28)] = 0
                x[np.random.randint(j,j+28)] = 0

            test.append(x)
            test_label.append(j)

        return train,test,test_label

    def generate_2d(self):
        pass
