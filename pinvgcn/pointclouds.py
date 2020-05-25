
import numpy as np
import os
from zipfile import ZipFile


import torch
from torch_geometric.data import download_url
import fastadj

from .data import setup_spectral_data, SingleSliceDataset


def load_point_cloud_data(name, dir=None):
    r"""
    Load a point cloud dataset and return its Data object. Currently
    supported dataset names are Oakland1 and Oakland2. Upon first usage, data
    is downloaded from 
    [http://www.cs.cmu.edu/~vmr/datasets/oakland_3d/cvpr09/doc/].
    """
    
    if dir is None:
        dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data')
    path = os.path.join(dir, name)
    
    if name == 'Oakland1':
        data = OaklandDataset(path, 1).data
    elif name == 'Oakland2':
        data = OaklandDataset(path, 2).data
    else:
        raise ValueError('Unknown point cloud dataset name: ' + name)
    
    data.name = name
    return data



class PointCloudSpectralSetup(object):
    r"""
    Class in the style of a torch_geometric transform. Augments a data object
    with spectral information on the Laplacian of the fully connected graph 
    with Gaussian edge weights. The Gaussian shape is determined by the
    parameter `sigma`. `eig_tol` is the tolerance for the 
    eigenvalue computation. `eig_threshold` determines which eigenvalues are 
    treated as zero. `fastadj_setup_name` is a string constant determining the
    accuracy of the adjacency approximation (cf. the `fastadj` module,
    possible values are `'default'`, `'rough'`, and `'fine'`).
    """
    
    def __init__(self, sigma, rank, eig_tol=1e-3, eig_threshold=1e-2, fastadj_setup_name='default'):
        self.sigma = sigma
        self.rank = rank
        self.eig_tol = eig_tol
        self.eig_threshold = eig_threshold
        self.fastadj_setup_name = fastadj_setup_name

    def __call__(self, data):
        w, U = point_cloud_laplacian_decomposition(data.x.cpu().numpy(), self.sigma, self.rank+1,
                                                   tol = self.eig_tol, fastadj_setup_name = self.fastadj_setup_name)
        
        setup_spectral_data(data, w, U, threshold=self.eig_threshold)

        return data



class OaklandDataset(SingleSliceDataset):
    r"""Subclass of InMemoryDataset that downloads and processes one of two
    possible subsets of the Oakland dataset from 
    [http://www.cs.cmu.edu/~vmr/datasets/oakland_3d/cvpr09/doc/].
    """
    
    def __init__(self, root, index, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        
        self.index = index
    
    url_base = 'http://www.cs.cmu.edu/~vmr/datasets/oakland_3d/cvpr09/data/'
    zip_names = {
        1: 'training.zip',
        2: 'validation.zip'
    }
    member_names = {
        1: 'training\oakland_part3_an_training.xyz_label_conf',
        2: 'validation\oakland_part3_am.xyz_label_conf'
    }
    
    @property
    def raw_file_names(self):
        return 'points.data'
    
    def download(self):
        zip_name = self.zip_names[self.index]
        download_url(self.url_base + zip_name, self.raw_dir)
        with ZipFile(os.path.join(self.raw_dir, zip_name), 'r') as file:
            file.extract(self.member_names[self.index], self.raw_paths[0])


    def process(self):
        table = np.genfromtxt(self.raw_paths[0])
        x = table[:,:3]
        y = np.unique(table[:,3], return_inverse=True)[1]
        
        self.save_processed(
            x = torch.tensor(x, dtype=torch.float), 
            y = torch.tensor(y, dtype=torch.long),
            num_classes = y.max()+1)



def point_cloud_laplacian_decomposition(x, sigma, num_ev, tol=1e-3, fastadj_setup_name='default'):
    r"""Return a partial eigen decomposition of the Laplacian of a fully 
    connected graph with Gaussian edge weights. The computation is performed
    by the `fastadj` module.
    """
    
    adj = fastadj.fixed_adjacency_matrix(x, sigma, setup=fastadj_setup_name)

    w, U = fastadj.normalized_adjacency_eigs(adj, k=num_ev, tol=tol)
    w = 1 - w
    
    return w, U
