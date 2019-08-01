from environments import MNIST, SMNIST, LMNIST, GYM

env_opts = {
    'MNIST': MNIST,
    'SMNIST': SMNIST,
    'LMNIST': LMNIST,
    'GYM': GYM
}

class EnvManager:
    def __init__(self, args):
        self.args = args
        # create environment
        self.env = env_opts[args['env']](args)

    def reset(self, **kwargs):
        return self.env.reset(kwargs)

    def step(self, action, **kwargs):
        return self.env.step(action, kwargs)

    def eval(self, action):
        return self.env.eval(action)
