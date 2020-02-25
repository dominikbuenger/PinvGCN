
import numpy as np
import scipy.sparse as sp

from warnings import warn

from .pinvdata import apply_pinv_filter, ScalingPlusLowRankPinvData, LowRankPinvData


def normalized_incidence(data):
    # if 'hyperedge_index' in data:
    inc = data.x.cpu().numpy()
    if 'hyperedge_weight' in data:
        weights = data.hyperedge_weight.cpu().numpy()
    else:
        weights = np.ones(inc.shape[1], dtype=np.float)
    
    dV = inc @ weights
    dV = 1/np.sqrt(dV[:,np.newaxis])
    dE = np.ones(inc.shape[0]) @ inc
    dE = np.sqrt(weights / dE)
    return dV * inc * dE

def hypergraph_laplacian_decomposition(inc, num_ev=None, tol=0):
    if num_ev is None or num_ev > inc.shape[1]/2:
        U, sigma, _ = np.linalg.svd(inc, full_matrices=False)
        if num_ev is not None:
            U = U[:,:num_ev]
            sigma = sigma[:num_ev]
    else:
        U, sigma, _ = sp.linalg.svds(inc, num_ev, tol=tol)
        
    return 1 - sigma.astype(np.float32)**2, U.astype(np.float32)


class HypergraphPinv(object):
    
    def __init__(self, rank=None, eig_tol=0, eig_threshold=1e-6):
        self.rank = rank
        self.eig_tol = eig_tol
        self.eig_threshold = eig_threshold
        
    def __call__(self, data):
        inc = normalized_incidence(data)
    
        if self.rank is None:
            w, U = hypergraph_laplacian_decomposition(inc, tol=self.eig_tol)
            w, U, alpha = apply_pinv_filter(w, U, threshold=self.eig_threshold)
            
            data = ScalingPlusLowRankPinvData(alpha, w-alpha, U, data)
        
        else:
            w, U = hypergraph_laplacian_decomposition(inc, self.rank+1, tol=self.eig_tol)
            w, U, _ = apply_pinv_filter(w, U, threshold=self.eig_threshold)
            
            if w.size != self.rank:
                warn('Rank {} does not match target rank {} (multiplicity of EV zero is {} instead of 1)'.format(
                    w.size, self.rank, self.rank+1-w.size))
            
            data = LowRankPinvData(w, U, data)

        return data