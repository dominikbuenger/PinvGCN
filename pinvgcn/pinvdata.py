
import numpy as np

import torch
import torch_geometric


def apply_pinv_filter(w, U, threshold = 1e-4, max_rank=None, keep_zero=False, beta=0.0):
    r"""Transform a given (partial) eigen decomposition of the Laplacian into
    its pseudoinverse. Eigenvalues lower than the threshold are treated as
    zero. If max_rank is given, only that many nonzero eigenvalues are used.
    If keep_zero is True, the zero eigenvalues will be maintained in the
    output, otherwise only the subset of nonzero eigenvalues and corresponding
    eigenvectors is returned (this is the default behaviour). The return value
    is a tuple of the scaled pseudoinverse eigenvalues, corresponding 
    eigenvector matrix, and the original smallest nonzero eigenvalue (which 
    was already used to scale the eigenvalues.)
    """
    mask = w > threshold
    if not any(mask):
        raise ValueError("Did not find a single positive eigenvalue out of {} (max: {:.4e})".format(w.size, w.max()))
    
    if keep_zero or beta != 0:
        alpha = w[mask].min()
        w[mask] = alpha / w[mask]
        w[~mask] = beta
        if max_rank is not None and max_rank < mask.sum():
            ind = np.argpartition(w, -max_rank)[:-max_rank]
            w[ind] = 0
    else:
        w = w[mask]
        alpha = w.min()
        w = alpha / w
        U = U[:,mask]
        if max_rank is not None and max_rank < w.size:
            ind = np.argpartition(w, -max_rank)[-max_rank:]
            w = w[ind]
            U = U[:, ind]
    
    return w, U, alpha


class BasePinvData(torch_geometric.data.Data):
    
    def __init__(self, base_data=None, **kwargs):
        if base_data is not None:
            super().__init__(**{key: val for key, val in base_data}, **kwargs)
        else:
            super().__init__(**kwargs)
        return self
    
    def random_split(self, s=None):
        if s is None:
            return
        
        device = self.y.device
        y = self.y.cpu().numpy()
        self.train_mask = torch.zeros(self.num_nodes, dtype=bool, device=device)
        for c in range(self.num_classes):
            ind = np.nonzero(y == c)[0]
            ind = np.random.choice(ind, s, replace=False)
            self.train_mask[ind] = True
        self.test_mask = ~self.train_mask
        
    def prepare_training(self):
        pass

    def apply_pinv(self, x, W=None, training=False):
        return None
    

class LowRankPinvData(BasePinvData):
    r"""Data subclass which stores a low-rank approximation to the Laplacian
    pseudoinverse and provides the apply_pinv method to perform multiplications
    with it."""
    
    def __init__(self, w, U, base_data=None, **kwargs):
        super().__init__(base_data, **kwargs)
        
        self.pinv_w = torch.FloatTensor(w[:,np.newaxis])
        self.pinv_U = torch.FloatTensor(U)
        
        self.pinv_U_training = None
        self.pinv_preconvolved_x = None
    
    def prepare_training(self):
        r"""Setup training by precomputing the convolution of data.x with the
        pseudoinverse and setting up submatrices required for efficient 
        computation of only the training rows of the forward operation output.
        """
        self.pinv_U_training = self.pinv_U[self.train_mask, :]
        
        if 'x' not in self:
            self.pinv_preconvolved_x = torch.matmul(self.pinv_U, torch.mul(self.pinv_w, self.pinv_U.t()))
        else:
            self.pinv_preconvolved_x = self.apply_pinv(self.x)
            
    def apply_pinv(self, X, W=None, training=False):
        r"""Compute U*diag(w)*U.T*X*W, where w and U are stored in the 
        pinv_w and pinv_U fields of this data object. W may be None to omit
        multiplication with weights. If training is True, the function will
        only return a subset of rows of the output corresponding to the
        training samples."""
        if training and self.pinv_U_training is None:
            raise "data.prepare_training must be called before training"
        
        X = torch.matmul(self.pinv_U.t(), X)
        if W is not None:
            X = torch.matmul(X, W)
        X = torch.mul(self.pinv_w, X)
        return torch.matmul(self.pinv_U_training if training else self.pinv_U, X)
        
        
class ScalingPlusLowRankPinvData(BasePinvData):
    r"""Data subclass which stores a representation of the Laplacian
    pseudoinverse as a scaled identity plus a low-rank matrix and provides the 
    apply_pinv method to perform multiplications with it."""
    
    def __init__(self, alpha, w, U, base_data=None, **kwargs):
        super().__init__(base_data, **kwargs)
        
        self.pinv_alpha = torch.FloatTensor([alpha])
        self.pinv_w = torch.FloatTensor(w[:,np.newaxis])
        self.pinv_U = torch.FloatTensor(U)
        
        self.pinv_U_training = None
        self.pinv_preconvolved_x = None
        
    
    def prepare_training(self):
        r"""Setup training by precomputing the convolution of data.x with the
        pseudoinverse and setting up submatrices required for efficient 
        computation of only the training rows of the forward operation output.
        """
        self.pinv_U_training = self.pinv_U[self.train_mask, :]
        
        # if 'x' not in self:
        #     self.pinv_preconvolved_x = torch.matmul(self.pinv_U, torch.mul(self.pinv_w, self.pinv_U.t())) + self.pinv_alpha*torch.eye(self.num_nodes, self.num_nodes, device=self.alpha.device, dtype=torch.float)
        # else:
        self.pinv_preconvolved_x = self.apply_pinv(self.x)
        
    def apply_pinv(self, X, W=None, training=False):
        r"""Compute (a*I+U*diag(w)*U.T)*X*W, where a, w, and U are stored in
        the pinv_alpha, pinv_w, and pinv_U fields of this data object. W may 
        be None to omit multiplication with weights. If training is True, the 
        function will only return a subset of rows of the output corresponding
        to the training samples."""
        if training:
            if self.pinv_U_training is None:
                raise "data.prepare_training must be called before training"
            X1 = X[self.train_mask,:]
            X2 = torch.matmul(self.pinv_U.t(), X)
            if W is not None:
                X1 = torch.matmul(X1, W)
                X2 = torch.matmul(X2, W)
        else:
            X1 = torch.matmul(X, W) if W is not None else X
            X2 = torch.matmul(self.pinv_U.t(), X1)

        X1 = self.pinv_alpha * X1
        X2 = torch.mul(self.pinv_w, X2)
        X2 = torch.matmul(self.pinv_U_training if training else self.pinv_U, X2)
        return X1 + X2
        
        