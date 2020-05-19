
import numpy as np
import os
from queue import Queue

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid


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
        if 'train_mask' in data:
            if 'val_mask' not in data:
                data.val_mask = torch.zeros(data.num_nodes, dtype=bool)
    
            if 'test_mask' not in data:
                data.test_mask = ~data.train_mask & ~data.val_mask
    
        elif 'train_idx' in data:
            def idx_to_mask(idx):
                mask = torch.zeros(data.num_nodes, dtype=torch.bool)
                mask[idx] = True
                return mask
    
            data.train_mask = idx_to_mask(data.train_idx)
    
            if 'val_idx' in data:
                data.val_mask = idx_to_mask(data.val_idx)
            else:
                data.val_mask = torch.zeros(data.num_nodes, dtype=bool)
    
            if 'test_idx' in data:
                data.test_mask = idx_to_mask(data.test_idx)
            else:
                data.test_mask = ~data.train_mask & ~data.val_mask
    
            data.train_idx = data.test_idx = data.val_idx = None
        else:
            data.train_mask = data.test_mask = data.val_mask = None
            
        
        if 'edge_weight' not in data:
            if 'edge_attr' in data:
                data.edge_weight = data.edge_attr
            else:
                data.edge_weight = None
            
        if self.lcc:
            data = self.lcc_data(data)
        return data
    
    
    def lcc_data(self, data):
        edge_index = data.edge_index.cpu().numpy()
        node_mask = self.lcc_mask(edge_index, data.num_nodes)
            
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

    def lcc_mask(self, edge_index, num_nodes):
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