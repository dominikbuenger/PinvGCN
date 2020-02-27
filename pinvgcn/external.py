
from .pinvdata import apply_pinv_filter, LowRankPinvData

from scipy.io import loadmat
import os

import torch


def load_from_matlab(dataset, filename, rank=None):
    r"""Load x and y data and Laplacian eigeninformation computed in MATLAB.
    The file must be located in data/{dataset}/{filename}. If rank is not 
    None, only that many eigenvalues from the stored decomposition are used.
    Returns a data object that provides multiplication with the pseudoinverse.
    """
    matfile = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', dataset, filename)
    try:
        matdata = loadmat(matfile)
        w = matdata['w'].flatten()
        U = matdata['U']
        x = matdata['x']
        y = matdata['y'].flatten()-1
    except:
        raise ValueError('Error reading MATLAB file ' + matfile) from None
    
    w, U, _ = apply_pinv_filter(w, U, threshold=1e-4, max_rank=rank)
    
    return LowRankPinvData(w, U, x=torch.FloatTensor(x), y=torch.LongTensor(y), num_classes=y.max()+1)