
import numpy as np
import os

import torch
from torch_geometric.data import Data, InMemoryDataset, download_url



def load_hypergraph_data(name, dir=None):
    r"""Load a hypergraph dataset and return its Data object. Currently
    supported dataset names are Mushroom, Covertype45, and Covertype67. Upon
    first usage, data is downloaded from the UCI website and then processed to
    turn the categorial attributes into hyperedges."""
    if dir is None:
        dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data')
    path = os.path.join(dir, name)
    
    if name == 'Mushroom':
        data = MushroomDataset(path).data
        data.num_classes = 2
    elif name == 'Covertype45':
        data = CovertypeDataset(path, [4,5]).data
        data.num_classes = 2
    elif name == 'Covertype67':
        data = CovertypeDataset(path, [6,7]).data
        data.num_classes = 2
    else:
        raise ValueError('Unknown hypergraph dataset name: ' + name)
    
    data.name = name
    return data


class UCIHypergraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def processed_file_names(self):
        return 'data.pt'

    def save_processed(self, incidence, labels):
        data = Data(
            x = torch.FloatTensor(incidence),
            y = torch.LongTensor(labels),
            hyperedge_weight = torch.ones(incidence.shape[1], dtype=torch.float))
        
        if self.pre_transform is None:
            data = self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])


class MushroomDataset(UCIHypergraphDataset):
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
    
        self.save_processed(incidence, labels)
        


class CovertypeDataset(UCIHypergraphDataset):
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
            
        self.save_processed(incidence, labels)