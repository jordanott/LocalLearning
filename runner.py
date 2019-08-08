import sherpa
import sherpa.schedulers
import argparse
import itertools

parser = argparse.ArgumentParser()

# sherpa params
parser.add_argument('--gpus',default='0,1,2,3',type=str)
parser.add_argument('--max_concurrent', type=int, default=1)

# params okay left as defaults
parser.add_argument('--max_layers', type=int, default=5, help='Max dense layers allowed')
parser.add_argument('--epochs', type=int, default=25, help='Number of epochs used for training')

parser.add_argument('--neurons', type=int, default=10, help='Total number of neurons is this squared')
parser.add_argument('--vis_weights', default=False, action='store_true', help='Store visualizations of weights')
parser.add_argument('--gif_weights', default=False, action='store_true', help='Store gif visualizations of weights')
parser.add_argument('--env', type=str, default='MNIST', choices=['MNIST', 'SMNIST', 'LMNIST', 'GYM'], help='Type of task')
FLAGS = parser.parse_args()

#Define Hyperparameter ranges
parameters = [
    sherpa.Ordinal('record', ['AND', 'OR']),
    sherpa.Discrete('num_layers', [1, FLAGS.max_layers])
]

for i in range(1,FLAGS.max_layers+1):
    parameters.extend([
        sherpa.Continuous('learning_rate_{}'.format(i), [0.000001, 0.5]),
        sherpa.Ordinal('INTRA_ON_{}'.format(i), [1, 0]),
        sherpa.Ordinal('APICAL_ON_{}'.format(i), [1, 0]),
        sherpa.Continuous('sparsity_{}'.format(i), [0.05, 0.5]),
        sherpa.Continuous('activity_decay_{}'.format(i), [0.05, 0.99]),
        sherpa.Continuous('threshold_{}'.format(i), [0.05, 0.99]),
        sherpa.Continuous('reset_potential_{}'.format(i), [0., 0.99]),
    ])

dict_flags = vars(FLAGS)

for arg in dict_flags:
    parameters.append(sherpa.Choice(name=arg, range=[dict_flags[arg] ]))

algorithm = sherpa.algorithms.RandomSearch(max_num_trials=500)
# gpus = [int(x) for x in FLAGS.gpus.split(',')]
# processes_per_gpu = FLAGS.max_concurrent//len(gpus)
# assert FLAGS.max_concurrent%len(gpus) == 0
# resources = list(itertools.chain.from_iterable(itertools.repeat(x, processes_per_gpu) for x in gpus))
scheduler = sherpa.schedulers.LocalScheduler() # resources=resources

# build_directory('SherpaResults/output/')

# Running it all
sherpa.optimize(algorithm=algorithm,
                scheduler=scheduler,
                parameters=parameters,
                lower_is_better=True,
                filename='main.py',
                max_concurrent=FLAGS.max_concurrent,
                output_dir='output_path')
