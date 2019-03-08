from environments import MNIST, LMNIST, GYM

env_opts = {
    'MNIST': MNIST,
    'LMNIST': LMNIST,
    'GYM': GYM
}

class EnvManager:
    def __init__(self, args):
        self.args = args
        # create environment
        self.env = env_opts[args['env']](args)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)
