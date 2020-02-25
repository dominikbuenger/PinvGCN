
import argparse
import os
import torch
import numpy as np
from time import perf_counter as timer

from pinvgcn.graphs import SBMData
from pinvgcn.model import PinvGCN
from pinvgcn.graphpinv import GraphPinv
from pinvgcn.summary import print_results, save_results
from pinvgcn.seeds import set_seed

def positive_or_none(string):
    val = int(string)
    return val if val > 0 else None

parser = argparse.ArgumentParser(description='Perform several runs of training and testing a Pseudoinverse GCN.')

parser.add_argument('p', type=float,
    help='the edge probability for nodes from the same class. If >1, this is divided by the number of samples per class')
parser.add_argument('q', type=float,
    help='the edge probability for nodes from different classes. If >1, this is divided by the number of samples per class')
parser.add_argument('num_classes', type=int, nargs='?', default=2, metavar='classes',
    help='the number of classes [2]')
parser.add_argument('block_size', type=int, nargs='?', default=500, metavar='block-size',
    help='the number of samples per class [500]')
parser.add_argument('split_size', type=float, nargs='?', default=5.0, metavar='split-size',
    help='the number of training samples per class, either as an absolute number or a ratio [5]')

parser.add_argument('-n', '--num-runs', default=1, type=int, metavar='N',
    help='number of runs to be performed')
parser.add_argument('-r', '--rank', type=positive_or_none, default=None, metavar='R',
    help='perform a low-rank approximation with this target rank')
parser.add_argument('-l', '--loops', type=float, default=0.0, metavar='WEIGHT',
    help='Add self loop edges with the given weight. If loops are already present, their weight is increased')
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

c = args.num_classes
b = args.block_size
n = c*b
p = args.p/n if args.p > 1 else args.p
q = args.q/n if args.q > 1 else args.q
s = int(args.split_size*b if args.split_size < 1 else args.split_size)


sbm = SBMData(p, q, c, b, s)
sbm.to(device)

tic = timer()
pinv_transform = GraphPinv(rank=args.rank, loop_weights=args.loops, dense_graph=True)

model = PinvGCN(n, c, hidden=args.hidden, dropout=args.dropout, bias=not args.no_bias)
model.to(device)

optimizer = torch.optim.Adam([
        dict(params=model.reg_params(), weight_decay=5e-4),
        dict(params=model.non_reg_params(), weight_decay=0)], lr=0.01)
setup_time_model = timer() - tic
print('Model setup done in {:.4} seconds'.format(setup_time_model))

setup_times = []
training_times = []
accuracies = []


for run in range(args.num_runs):
    if args.no_fixed_seeds:
        set_seed(run)

    sbm.generate_adjacency()

    tic = timer()
    data = pinv_transform(sbm)
    data.to(device)
    t1 = timer() - tic
    setup_times.append(t1)

    tic = timer()
    model.reset_parameters()
    model.run_training(data, optimizer, num_epochs=500)
    t2 = timer() - tic
    training_times.append(t2)

    acc = model.eval_accuracy(data)
    accuracies.append(acc)
    print('Run {}/{}: Setup time {:.4f} s, Training time {:.4f} s, accuracy {:.4f} %'.format(
        run+1, args.num_runs, t1, t2, 100*acc))

setup_times = np.add(setup_time_model, setup_times)

print('###')
print_results(accuracies, setup_times, training_times)
print('###')

if not args.no_save:
    dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results', 'sbm')
    dataset_name = data.name
    
    architecture_name = 'PinvGCN'
    if args.rank is not None:
        architecture_name += '-rank{}'.format(args.rank)
    if args.no_bias:
        architecture_name += '-nobias'
    if args.loops > 0:
        architecture_name += '-loops{}'.format(args.loops)
    
    save_results(
        dir, dataset_name, architecture_name,
        accuracies, setup_times, training_times,
        args.__dict__, file=__file__)
    
