
import argparse
import numpy as np
import os
import torch
from time import perf_counter as timer

import pinvgcn

### PARSE ARGUMENTS

parser = argparse.ArgumentParser(description='Perform several runs of training and testing a Pseudoinverse GCN on a graph dataset.')

parser.add_argument('dataset', help='Name of the dataset to be loaded')
parser.add_argument('sigma', type=float, help='Gaussian shape parameter')
parser.add_argument('coefficients', help='Name of the coefficient setup')

parser.add_argument('-a', '--alpha', type=float, default=1, help='Filter shape parameter. Default: 1')
parser.add_argument('-b', '--beta', type=float, default=1, help='Filter shape parameter. Default: 1')
parser.add_argument('-c', '--gamma', type=float, default=1, help='Filter shape parameter. Default: 1')

parser.add_argument('-n', '--num-runs', default=1, type=int, metavar='N',
    help='Number of runs to be performed')
parser.add_argument('-s', '--split-size', type=int, default=None, metavar='S',
    help="Randomly split the nodes into training and test set using S training samples per class")
parser.add_argument('-r', '--rank', type=int, default=None, metavar='R',
    help='Perform a low-rank approximation with this target rank')
parser.add_argument('-l', '--loops', type=float, default=0.0, metavar='WEIGHT',
    help='Add self loop edges with the given weight. If loops are already present, their weight is increased')
parser.add_argument('--hidden', nargs='*', type=int, default=[32], metavar='H', 
    help='Hidden layer widths')
parser.add_argument('--dropout', type=float, default=0.5, metavar='RATE',
    help='Dropout rate')
parser.add_argument('--learning-rate', default=0.05, type=float, metavar='LR',
    help='Learning rate')
parser.add_argument('--weight-decay', default=5e-4, type=float, metavar='DECAY',
    help='Weight decay on weight matrices (not on bias vectors)')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
    help='Number of training epochs per run')
parser.add_argument('--no-bias', default=False, action='store_true',
    help='Disable bias')
parser.add_argument('--disable-cuda', action='store_true', default=False,
    help='Disable CUDA acceleration')
parser.add_argument('--no-save', action='store_true', default=False,
    help='Disable saving results')
parser.add_argument('--no-fixed-seeds', action='store_true', default=False,
    help='Disable fixed seeds. Also implies that the results will not be saved')
parser.add_argument('--run-print-level', default='results',
    help='Level of information that is printed after each run. Possible values are "none", "results", and "weights"')
parser.add_argument('--print-weights', action='store_const', const='weights', dest='run_print_level',
    help='Equivalent to "--run-print-level weights"')


args = parser.parse_args()


### PREPARATIONS

if args.no_fixed_seeds and not args.no_save:
    print('no-fixed-seeds option given. Results will not be reproducible and hence not be saved.')
    args.no_save = True

base_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(base_dir, 'data')

if not args.disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


### SETUP

data = pinvgcn.load_point_cloud_data(args.dataset, data_dir)

tic = timer()

data = pinvgcn.PointCloudSpectralSetup(args.sigma, args.rank)(data)
data = data.to(device)

coeffs = pinvgcn.get_coefficient_preset(args.coefficients, alpha=args.alpha, beta=args.beta, gamma=args.gamma)
pinvgcn.print_filter_values(coeffs, data)

model = pinvgcn.PinvGCN(coeffs, data.num_features, data.num_classes, hidden_channels=args.hidden, 
                        dropout=args.dropout, bias=not args.no_bias)
model.to(device)

optimizer = torch.optim.Adam(model.split_parameters(args.weight_decay), lr=args.learning_rate)

setup_time = timer() - tic
print('Setup done in {:.4} seconds'.format(setup_time))


### EXPERIMENT RUNS

training_times = []
accuracies = []

for run in range(args.num_runs):
    if not args.no_fixed_seeds:
        pinvgcn.set_seed(run)
    
    pinvgcn.random_split(data, args.split_size)

    tic = timer()
    
    model.reset_parameters()
    
    input = model.preconvolve_input(data, data.x)
    
    model.run_training(data, input, optimizer, num_epochs=args.epochs)
    
    t = timer() - tic
    training_times.append(t)

    acc = model.eval_accuracy(data, input)
    accuracies.append(acc)
    
    if args.run_print_level == 'results':
        print('Run {: 4d}/{}: Training time {:.4f} s, accuracy {:.4f} %'.format(run+1, args.num_runs, t, 100*acc))
    elif args.run_print_level == 'weights':
        with np.printoptions(precision=3, suppress=True):
            print('Run {: 4d}/{}: Training time {:.4f} s, accuracy {:.4f} %, avg. abs. weights:'.format(run+1, args.num_runs, t, 100*acc), 
                ", ".join("L{} {}".format(i+1, ww) for i, ww in enumerate(model.average_absolute_weight_entries())))

print('###')
pinvgcn.print_results(accuracies, setup_time, training_times)
print('###')


### SAVE RESULTS

if not args.no_save:
    results_dir = os.path.join(base_dir, 'results', 'pointclouds')
    
    dataset_name = (data.name if 'name' in data else args.dataset) + '_sigma' + str(args.sigma)
    if args.split_size is not None:
        dataset_name += '_split{}'.format(args.split_size)
    
    architecture_name = 'PinvGCN_' + args.coefficients
    for p in ['alpha','beta','gamma']:
        val = getattr(args, p)
        if val != 1:
            architecture_name += '_{}{}'.format(p,val)
    if args.rank is not None:
        architecture_name += '_rank{}'.format(args.rank)
    if args.no_bias:
        architecture_name += '_nobias'
    if args.loops > 0:
        architecture_name += '_loops{}'.format(args.loops)
    
    pinvgcn.save_results(
        results_dir, dataset_name, architecture_name,
        accuracies, setup_time, training_times,
        args.__dict__, file=__file__)
    