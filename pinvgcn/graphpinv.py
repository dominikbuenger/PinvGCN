

import numpy as np
import scipy.sparse as sp

from warnings import warn

from .pinvdata import apply_pinv_filter, LowRankPinvData


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

class GraphPinv(object):
    r"""Class in the style of a torch_geometric transform. Replaces given data 
    by a new data object holding a representation of the graph Laplacian
    and providing the apply_pinv method. If rank is not None, a low-rank
    approximation is used. eig_tol is the tolerance for the SVD. eig_threshold
    determines which eigenvalues are treated as zero. If loop_weights is not
    None, additional self loops are added with that weight. If dense_graph is
    True, internal computations are done with the adjacency matrix stored as
    a numpy array instead of a scipy.sparse.coo_matrix."""
    
    
    def __init__(self, rank=None, loop_weights=None, dense_graph=False, eig_tol=0, eig_threshold=1e-6):
        self.rank = rank
        self.loop_weights = loop_weights
        self.dense_graph = dense_graph
        self.eig_tol = eig_tol
        self.eig_threshold = eig_threshold
        
    def __call__(self, data):
        r = self.rank
        adj = normalized_adjacency(data, self.loop_weights, self.dense_graph)
        
        if r is None:
            raise NotImplementedError("Full non-approximated Pseudoinverse is not available for graphs")
        else:
            w, U = graph_laplacian_decomposition(adj, self.rank+1, tol=self.eig_tol)
            w, U, _ = apply_pinv_filter(w, U, threshold=self.eig_threshold)
            
            if w.size != self.rank:
                warn('Rank {} does not match target rank {} (multiplicity of EV zero is {} instead of 1)'.format(
                    w.size, self.rank, self.rank+1-w.size))
            
            data = LowRankPinvData(w, U, data)

        return data