

class EnvManager:
    def __init__(self, args):
        self.args = args
        self.args['input_shape'] = (28,28)
