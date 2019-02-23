
class Environment:
    def __init__(self, args):
        self.args = args

    def step(self):
        pass

    def act(self):
        pass



class MNIST(Environment):
    def __init__(self, args):
        super(MNIST, self).__init__()


class LMNIST(Environment):
    def __init__(self, args):
        super(LMNIST, self).__init__()


class GYM(Environment):
    def __init__(self, args):
        super(GYM, self).__init__()
