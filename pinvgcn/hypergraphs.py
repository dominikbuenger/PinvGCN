
import numpy as np
import scipy.sparse as sp
import os

import torch
from torch_geometric.data import download_url

from .data import setup_spectral_data, SingleSliceDataset


def load_hypergraph_data(name, dir):
    r"""Load a hypergraph dataset and return its Data object. Currently
    supported dataset names are Mushroom, Covertype45, and Covertype67. Upon
    first usage, data is downloaded from the UCI website and then processed to
    turn the categorial attributes into hyperedges."""
    
    path = os.path.join(dir, name)
    
    if name == 'Mushroom':
        data = MushroomDataset(path).data
    elif name == 'Covertype45':
        data = CovertypeDataset(path, [4,5]).data
    elif name == 'Covertype67':
        data = CovertypeDataset(path, [6,7]).data
    else:
        raise ValueError('Unknown hypergraph dataset name: ' + name)
    
    data.name = name
    return data



class HypergraphSpectralSetup(object):
    r"""
    Class in the style of a torch_geometric transform. Augments a data object
    with spectral information on the hypergraph Laplacian. If `rank` is not 
    None, a low-rank approximation is used. If `partial_eigs` is True,
    only the relevant eigenvalues are computed, which might negatively impact
    runtime for small systems. In the partial case, `eig_tol` is the tolerance
    for the eigenvalue computation. `eig_threshold` determines which 
    eigenvalues are treated as zero.
    This function currently expects the hypergraph incidence to be stored in
    data.x as a dense tensor.
    """
    
    def __init__(self, rank=None, eig_tol=0, eig_threshold=1e-6, partial_eigs=False):
        self.rank = rank
        self.eig_tol = eig_tol
        self.eig_threshold = eig_threshold
        self.partial_eigs = partial_eigs
        
    def __call__(self, data):
        inc = normalized_incidence(data)
    
        w, U = hypergraph_laplacian_decomposition(inc, num_ev=None if self.rank is None else self.rank+1, 
                                                  partial_svd=self.partial_eigs, partial_tol=self.eig_tol)
    
        setup_spectral_data(data, w, U, threshold=self.eig_threshold, max_rank=self.rank)
    
        return data


class HypergraphDataset(SingleSliceDataset):
    def save_hypergraph(self, incidence, labels, weights=None):
        self.save_processed(
            x = torch.tensor(incidence, dtype=torch.float),
            y = torch.tensor(labels, dtype=torch.long),
            hyperedge_weight = torch.ones(incidence.shape[1], dtype=torch.float) if weights is None else \
                                torch.tensor(weights, dtype=torch.float))
        
class MushroomDataset(HypergraphDataset):
    r"""Subclass of InMemoryDataset that downloads and processes the Mushroom
    dataset from the UCI website."""
    
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'
    
    @property
    def raw_file_names(self):
        return 'agaricus-lepiota.data'
    
    def download(self):
        download_url(self.url, self.raw_dir)
        
    def process(self):
        raw = []
        with open(os.path.join(self.raw_dir, self.raw_file_names), 'r') as f:
            for line in f:
                raw.append(line[:-1].split(','))
                
        raw = np.array(raw)
        labels = (raw[:,0] == 'e').astype(int)
        
        inc_list = []
        for i in range(1, raw.shape[1]):
            col = raw[:,i]
            values = np.unique(col)
            if any([v.startswith('?') for v in values]):
                continue
            
            for v in values:
                edge = (col == v)
                print(' - Hyperedge size for attribute #{} == {}:  {}'.format(i, v, edge.sum()))
                inc_list.append(edge)
                
        incidence = np.array(inc_list, dtype=np.float).T
        print(' - Mushroom incidence shape: {}'.format(incidence.shape))
    
        self.save_hypergraph(incidence, labels)
        


class CovertypeDataset(HypergraphDataset):
    r"""Subclass of InMemoryDataset that downloads and processes the Covertype
    dataset from the UCI website. A subset of the data can be used by only 
    using certain classes."""
    
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz'
    
    def __init__(self, root, classes=None, num_bins=10, transform=None, pre_transform=None):
        self.classes = classes
        self.num_bins = num_bins
        super().__init__(root, transform, pre_transform)
    
    @property
    def raw_file_names(self):
        return 'covtype.data'
    
    def download(self):
        download_url(self.url, self.raw_dir)
        os.system('gunzip -f ' + os.path.join(self.raw_dir, self.raw_file_names + '.gz'))
        
    def process(self):
        raw_attributes, labels = [], []
        with open(os.path.join(self.raw_dir, self.raw_file_names), 'r') as f:
            for line in f:
                values = line.split(',')
                raw_attributes.append([int(v) for v in values[:-1]])
                labels.append(int(values[-1]))
        raw_attributes = np.array(raw_attributes)
        labels = np.array(labels)
        
        inc_list = []
        for attr in range(10):
            col = raw_attributes[:,attr]
            colmin = col.min()
            colmax = col.max()
            print(' - Continuous attribute #{}: min {}, max {}'.format(attr, colmin, colmax))
            
            b = colmin
            for i in range(self.num_bins):
                a = b
                b = colmax+1 if i==self.num_bins-1 else colmin + (colmax-colmin)*(i+1)/self.num_bins
                inc_list.append((col >= a) & (col < b))
        
        incidence = np.hstack((np.array(inc_list, dtype=int).T, raw_attributes[:,10:]))
        print(' - Full Covertype incidence shape: {}'.format(incidence.shape))
        
        if self.classes is not None:
            
            mask = [l in self.classes for l in labels]
            class_map = {orig: new for new, orig in enumerate(self.classes)}
            labels = [class_map[l] for l in labels[mask]]
            
            incidence = incidence[mask, :]
            edge_deg = incidence.sum(axis=0)
            mask = edge_deg > 0 # include hyperedges with a single node
            for attr in range(10):
                print(' - Hyperedges for continuous attribute #{}: {} out of {}'.format(attr, mask[attr*self.num_bins : (attr+1)*self.num_bins].sum(), self.num_bins))
            incidence = incidence[:, mask]
            
            print(' - Partial incidence shape for classes {}: {}'.format(self.classes, incidence.shape))
            
        self.save_hypergraph(incidence, labels)


def normalized_incidence(data):
    r"""Compute a numpy array holding D^{-1/2} H W^{1/2} B^{-1/2}, where H is 
    the hypergraph incidence matrix given by data.x, D is the diagonal node
    degree matrix, B is the diagonal hyperedge degree matrix, and W is the
    optional diagonal hyperedge weight matrix given by data.hyperedge_weight.
    """
    # TODO: check for other sources of incidence in data, e.g., hyperedge_index
    inc = data.x.cpu().numpy()
    if 'hyperedge_weight' in data:
        weights = data.hyperedge_weight.cpu().numpy()
    else:
        weights = np.ones(inc.shape[1], dtype=np.float)
    
    d = inc @ weights
    d = 1/np.sqrt(d[:,np.newaxis])
    b = np.ones(inc.shape[0]) @ inc
    b = np.sqrt(weights / b)
    return d * inc * b

def hypergraph_laplacian_decomposition(inc, num_ev=None, partial_svd=False, partial_tol=0):
    r"""Return a (partial) eigen decomposition of the hypergraph Laplacian. If
    num_ev is not None, only that many smallest eigenvalues are computed. If 
    partial_svd is True, only a partial SVD is computed via scipy.linalg.svds
    with tolerance given by the parameter partial_tol."""
    
    # TODO: get better heuristic of when to choose SVDS over SVD
    if partial_svd and num_ev is not None:
        U, sigma, _ = sp.linalg.svds(inc, num_ev, tol=partial_tol)
    else:
        U, sigma, _ = np.linalg.svd(inc, full_matrices=False)
        if num_ev is not None:
            U = U[:,:num_ev]
            sigma = sigma[:num_ev]
        
    return 1 - sigma.astype(np.float32)**2, U.astype(np.float32)
