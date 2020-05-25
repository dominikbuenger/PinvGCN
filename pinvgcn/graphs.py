
import numpy as np
import scipy.sparse as sp
import os
from queue import Queue
from warnings import warn

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid

from .data import setup_spectral_data, check_masks


def load_graph_data(name, dir=None, lcc=False):
    r"""Load a graph dataset and return its Data object."""
    full_name = name + '_LCC' if lcc else name
    if dir is None:
        dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data')
    path = os.path.join(dir, full_name)
    
    if name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(path, name, pre_transform=GraphPreprocess(lcc))
        data = dataset.data
        data.num_classes = dataset.num_classes
    else:
        raise ValueError("Unknown dataset: {}".format(name))
    
    data.name = full_name
    return data


class GraphPreprocess(object):
    r"""Class in the style of a torch_geometric transform. Used internally in
    load_graph_data to preprocess the data and optionally compute its largest
    connected component."""
    
    def __init__(self, lcc=False):
        self.lcc = lcc
    
    def __call__(self, data):
        
        check_masks(data)
        
        if 'edge_weight' not in data:
            data.edge_weight = data.edge_attr if 'edge_attr' in data else None
            
        if self.lcc:
            data = lcc_data(data)
            
        return data

class GraphSpectralSetup(object):
    r"""
    Class in the style of a torch_geometric transform. Augments a data 
    object with spectral information on the graph Laplacian. If `rank` is not 
    None, a low-rank approximation is used. eig_tol is the tolerance for the 
    eigenvalue computation. `eig_threshold` determines which eigenvalues are 
    treated as zero. If loop_weights is not None, additional self loops are 
    added with that weight. If dense_graph is True, internal computations are 
    done with the adjacency matrix stored as a numpy array instead of a 
    scipy.sparse.coo_matrix.
    """
    
    
    def __init__(self, rank=None, loop_weights=None, dense_graph=False, eig_tol=0, eig_threshold=1e-6):
        self.rank = rank
        self.loop_weights = loop_weights
        self.dense_graph = dense_graph
        self.eig_tol = eig_tol
        self.eig_threshold = eig_threshold
        
    def __call__(self, data):
        adj = normalized_adjacency(data, self.loop_weights, self.dense_graph)
        
        if self.rank is None:
            raise NotImplementedError("Full non-approximated Pseudoinverse is not available for graphs")
        else:
            w, U = graph_laplacian_decomposition(adj, self.rank+1, tol=self.eig_tol)
            setup_spectral_data(data, w, U, threshold=self.eig_threshold, max_rank = self.rank)
            
            if data.rank != self.rank:
                warn('Computed rank {} does not match target rank {}'.format(data.rank, self.rank))
            
        return data
    
class SBMData(Data):
    r"""Data subclass for generation of Stochastic Blockmodel data. Creates b
    samples for each of c classes. The first s samples of each class are 
    training samples. Graph edges are not created until generate_adjacency is
    called."""
    def __init__(self, p, q, c, b, s, **kwargs):
        y = np.hstack([i*np.ones(b, dtype=int) for i in range(c)])
        train_mask = np.zeros(c*b, dtype=bool)
        for i in range(c):
            train_mask[i*b:i*b+s] = True
        
        super().__init__(
            p=p, q=q, num_classes=c, block_size=b, split_size=s, 
            y = torch.tensor(y),
            train_mask = torch.tensor(train_mask),
            test_mask =  torch.tensor(~train_mask),
            __num_nodes__ = c*b,
            name = 'SBM-p{}-q{}-c{}-b{}-s{}'.format(p, q, c, b, s),
            **kwargs)
        
    def generate_adjacency(self):
        r""" Create random undirected, unweighted edges based on the p and q
        parameters given to the Data constructor."""
        c = self.num_classes
        b = self.block_size
        
        edges = []
        for c1 in range(c):
            for i in range(c1*b, (c1+1)*b):
                for j in range(i+1, (c1+1)*b):
                    if np.random.rand() < self.p:
                        edges.append([i,j])
                        edges.append([j,i])
                for c2 in range(c1+1, c):
                    for j in range(c2*b, (c2+1)*b):
                        if np.random.rand() < self.q:
                            edges.append([i,j])
                            edges.append([j,i])
                            
        self.edge_index = torch.tensor(np.array(edges).T, device=self.y.device)
        

def lcc_data(data):
    edge_index = data.edge_index.cpu().numpy()
    node_mask = lcc_mask(edge_index, data.num_nodes)
        
    if node_mask is None:
        print('Largest connected component: Full graph')
        return data
    print('Largest connected component: {}/{} nodes'.format(node_mask.sum(), data.num_nodes))

    node_indices = -np.ones(data.num_nodes, dtype=int)
    node_indices[node_mask] = np.arange(node_mask.sum())
    edge_mask = node_mask[edge_index[0]]
    edge_index = node_indices[edge_index[:,edge_mask]]

    kwargs = {}
    for key, val in data.__dict__.items():
        if val is None:
            pass
        elif key == 'x':
            val = val[node_mask,:]
        elif key in ['y','train_mask','test_mask','val_mask']:
            val = val[node_mask]
        elif key == 'edge_index':
            val = torch.tensor(edge_index)
        elif key in ['edge_weight', 'edge_attr']:
            val = val[edge_mask]
        kwargs[key] = val
    return Data(**kwargs)

def lcc_mask(edge_index, num_nodes):
    edge_index = edge_index[:, np.argsort(edge_index[0])]
    neighborhoods = np.zeros(num_nodes+1, dtype=int)
    j = 0
    for e in range(edge_index.shape[1]):
        i = edge_index[0, e]
        while j < i:
            j += 1
            neighborhoods[j] = e
    while j < num_nodes:
        j += 1
        neighborhoods[j] = edge_index.shape[1]
    # now the edge indices e with edge_index[0,e] == j are exactly those in range(neighborhoods[j], neighborhoods[j+1])
    
    components = np.zeros(num_nodes, dtype=int)
    c = 0
    q = Queue()
    for start in range(num_nodes):
        if components[start] != 0:
            continue
        c += 1
        components[start] = c
        q.put(start)
        while not q.empty():
            j = q.get()
            for k in edge_index[1, neighborhoods[j]:neighborhoods[j+1]]:
                if components[k] == 0:
                    components[k] = c
                    q.put(k)

    if c <= 1:
        return None
    c_max = 1+np.argmax([(components == i+1).sum() for i in range(c)])
    return components == c_max



def adjacency(data, loop_weights=None, dense=False):
    r"""Returns the adjacency matrix resulting from the edges given by the
    edge_index (and optionally edge_weight) fields in the data object. If 
    loop_weights is not None, additional self loops are added with that
    weight. By default, a scipy.sparse.coo_matrix is returned. If dense is 
    True, it is converted into a numpy array instead."""
    n = data.num_nodes
    ii, jj = data.edge_index.cpu().numpy()
    if 'edge_weight' not in data:
        ww = np.ones(data.num_edges, dtype=np.float32)
    else:
        ww = data.edge_weight.cpu().numpy()

    if loop_weights is not None and loop_weights != 0:
        ii = np.append(ii, np.arange(n))
        jj = np.append(jj, np.arange(n))
        ww = np.append(ww, loop_weights*np.ones(n))
    adj = sp.coo_matrix((ww, (ii,jj)), shape=(n,n))
    
    if dense:
        adj = adj.toarray()
    return adj

def normalized_adjacency(data, loop_weights=None, dense=False):
    r"""Returns the symmetrically normalized adjacency matrix resulting from 
    the edges given by the edge_index (and optionally edge_weight) fields in 
    the data object. The result is the unnormalized adjacency matrix 
    multiplied with the diagonal inverse square root degree matrix from both
    sides. If loop_weights is not None, additional self loops are added with 
    that weight. By default, a scipy.sparse.coo_matrix is returned. If dense is 
    True, it is converted into a numpy array instead.
    """
    adj = adjacency(data, loop_weights, dense)
    d = np.squeeze(np.asarray(adj.sum(1)))
    d = 1/np.sqrt(d)
    if dense:
        return d[:,np.newaxis] * adj * d
    else:
        return adj.multiply(d).multiply(d[:,np.newaxis]).tocsr()
    
    
def graph_laplacian_decomposition(adj, num_ev=None, tol=0):
    r"""Return a (partial) eigen decomposition of the graph Laplacian. If
    num_ev is not None, only that many smallest eigenvalues are computed. The 
    parameter tol is used for scipy.linalg.eigs (if it is called)."""
    n = adj.shape[0]
    if num_ev is None or num_ev > n/2:
        if sp.issparse(adj):
            adj = adj.toarray()
        w, U = np.linalg.eigh(adj)
        w = 1-w
        ind = np.argsort(w)
        if num_ev is not None:
            ind = ind[:num_ev]
        w = w[ind]
        U = U[:,ind]
    else:
        if sp.issparse(adj):
            adj = (adj + sp.identity(adj.shape[0])).tocsr()
        else:
            adj += np.identity(adj.shape[0])
        w, U = sp.linalg.eigsh(adj, num_ev, tol=tol)
        w = 2-w
    return w.astype(np.float32), U.astype(np.float32)
