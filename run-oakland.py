

import argparse
import os
import torch
from time import perf_counter as timer

from pinvgcn.external import load_from_matlab
from pinvgcn.model import PinvGCN
from pinvgcn.summary import print_results, save_results
from pinvgcn.seeds import set_seed

def positive_or_none(string):
    val = int(string)
    return val if val > 0 else None

parser = argparse.ArgumentParser(description='Perform several runs of training and testing a Pseudoinverse GCN on a graph dataset.')

parser.add_argument('dataset', help='the name of the dataset to be loaded')
parser.add_argument('-n', '--num-runs', default=1, type=int, metavar='N',
    help='number of runs to be performed')
parser.add_argument('-s', '--split-size', type=positive_or_none, default=None, metavar='S',
    help="randomly split the nodes into training and test set using S training samples per class")
parser.add_argument('-r', '--rank', type=positive_or_none, default=None, metavar='R',
    help='perform a low-rank approximation with this target rank')
parser.add_argument('--hidden', nargs='*', type=int, default=[32], metavar='H', 
    help='hidden layer widths')
parser.add_argument('--dropout', type=float, default=0.5, metavar='RATE',
    help='dropout rate')
parser.add_argument('--no-bias', default=False, action='store_true',
    help='disable bias')
parser.add_argument('--disable-cuda', action='store_true', default=False,
    help='disable CUDA acceleration')
parser.add_argument('--no-save', action='store_true', default=False,
    help='disable saving results')
parser.add_argument('--no-fixed-seeds', action='store_true', default=False,
    help='disable fixed seeds. Also implies that the results will not be saved.')

args = parser.parse_args()


if args.no_fixed_seeds and not args.no_save:
    print('no-fixed-seeds option given. Results will not be reproducible and hence not be saved.')
    args.no_save = True

if not args.disable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


tic = timer()
data = load_from_matlab('Oakland', args.dataset + '.mat', rank=args.rank)
data = data.to(device)
model = PinvGCN(data.num_features, data.num_classes, hidden=args.hidden, dropout=args.dropout, bias=not args.no_bias)
model.to(device)

optimizer = torch.optim.Adam([
        dict(params=model.reg_params(), weight_decay=5e-4),
        dict(params=model.non_reg_params(), weight_decay=0)], lr=0.01)
setup_time = timer() - tic
print('Loading from MAT file done in {:.4} seconds'.format(setup_time))

training_times = []
accuracies = []

for run in range(args.num_runs):
    if not args.no_fixed_seeds:
        set_seed(run)
    
    data.random_split(args.split_size)

    tic = timer()
    model.reset_parameters()
    model.run_training(data, optimizer, num_epochs=500)
    t = timer() - tic
    training_times.append(t)

    acc = model.eval_accuracy(data)
    accuracies.append(acc)
    print('Run {}/{}: Training time {:.4f} s, accuracy {:.4f} %'.format(
        run+1, args.num_runs, t, 100*acc))

print('###')
print_results(accuracies, setup_time, training_times)
print('###')


if not args.no_save:
    dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results', 'graphs')
    
    dataset_name = data.name if 'name' in data else args.dataset
    if args.split_size is not None:
        dataset_name += '-split{}'.format(args.split_size)
    
    architecture_name = 'PinvGCN'
    if args.rank is not None:
        architecture_name += '-rank{}'.format(args.rank)
    if args.no_bias:
        architecture_name += '-nobias'
    
    save_results(
        dir, dataset_name, architecture_name,
        accuracies, setup_time, training_times,
        args.__dict__, file=__file__)
    