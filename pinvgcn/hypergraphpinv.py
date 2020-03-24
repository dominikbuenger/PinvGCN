
import numpy as np
import scipy.sparse as sp

from warnings import warn

from .pinvdata import apply_pinv_filter, ScalingPlusLowRankPinvData, LowRankPinvData


def normalized_incidence(data):
    r"""Compute a numpy array holding D^{-1/2} H W^{1/2} D^{-1/2}, where H is 
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

def hypergraph_laplacian_decomposition(inc, num_ev=None, tol=0):
    r"""Return a (partial) eigen decomposition of the hypergraph Laplacian. If
    num_ev is not None, only that many smallest eigenvalues are computed. The 
    parameter tol is used for scipy.linalg.svds (if it is called)."""
    
    # TODO: get better heuristic of when to choose SVDS over SVD
    if num_ev is None or num_ev > inc.shape[1]/2:
        U, sigma, _ = np.linalg.svd(inc, full_matrices=False)
        if num_ev is not None:
            U = U[:,:num_ev]
            sigma = sigma[:num_ev]
    else:
        U, sigma, _ = sp.linalg.svds(inc, num_ev, tol=tol)
        
    return 1 - sigma.astype(np.float32)**2, U.astype(np.float32)


class HypergraphPinv(object):
    r"""Class in the style of a torch_geometric transform. Replaces given data 
    by a new data object holding a representation of the hypergraph Laplacian
    and providing the apply_pinv method. If rank is not None, a low-rank
    approximation is used. eig_tol is the tolerance for the SVD. eig_threshold
    determines which eigenvalues are treated as zero."""
    
    def __init__(self, rank=None, beta=0, eig_tol=0, eig_threshold=1e-6):
        self.rank = rank
        self.beta = beta
        self.eig_tol = eig_tol
        self.eig_threshold = eig_threshold
        
    def __call__(self, data):
        inc = normalized_incidence(data)
    
        if self.rank is None:
            w, U = hypergraph_laplacian_decomposition(inc, tol=self.eig_tol)
            w, U, alpha = apply_pinv_filter(w, U, threshold=self.eig_threshold, keep_zero=True, beta=self.beta)
            
            data = ScalingPlusLowRankPinvData(alpha, w-alpha, U, data)
        
        else:
            w, U = hypergraph_laplacian_decomposition(inc, self.rank+1, tol=self.eig_tol)
            w, U, _ = apply_pinv_filter(w, U, threshold=self.eig_threshold)
            
            if w.size != self.rank:
                warn('Rank {} does not match target rank {} (multiplicity of EV zero is {} instead of 1)'.format(
                    w.size, self.rank, self.rank+1-w.size))
            
            data = LowRankPinvData(w, U, data)

        return data