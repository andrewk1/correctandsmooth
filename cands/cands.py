"""
Correct&Smooth Implementation in numpy
"""
import torch
import torch_geometric

from functools import lru_cache
from tqdm import tqdm


def to_dense_adj(edge_index):
    """"""
    return torch_geometric.utils.to_dense_adj


@lru_cache(maxsize=32)
def normalize_adj_matrix(edge_index):
    """"""
    # TODO: Compute normalized adjancency matrix
    A = to_dense_adj(edge_index).squeeze()
    D = torch.diag(A.sum(-1))
    D_inv_sqrt = D.pow(-0.5)
    # Numerical errors from divide by 0, 
    # we follow the correction from the paper codebase at:
    # https://github.com/CUAI/CorrectAndSmooth/blob/b910314a59270984f5e249462ee3faa815fc9a0c/outcome_correlation.py#L77
    D_inv_sqrt[D_inv_sqrt == float('inf')] = 0 # 
    S = D_inv_sqrt @ A @ D_inv_sqrt
    return S


def smooth_operator(E, S, alpha = 0.8, eps = 1e-5, verbose=True):
    """
    E^(t+1) = (1-alpha1)E + alpha * S @ E^(t) -> Ehat
    """
    if verbose:
        pbar = tqdm(total=float('inf'))

    Ehat = E
    diff = eps
    itr = 0
    while diff >= eps:
        # This is the iterative update step
        Et = (1 - alpha) * E + alpha * (S @ Ehat)
        diff = float(torch.norm(Ehat - Et))
        Ehat = Et
        if verbose:
            pbar.update(1)
            pbar.set_postfix({ 'diff': diff })
        itr += 1
    return Ehat


def residual_error(y, yhat, val_index):
    """
    Form E, residual error matrix Z - L for training data"
    """
    E = yhat - y
    E[val_index] = 0
    return E


def autoscale(E, Ehat, Z, train_split_idxs):
    """
    sigma = sum of absolute value of E for each training sample / num training samples
    """
    sigma = float(sum(torch.norm(E[train_split_idxs], p=1, dim=-1))) / len(train_split_idxs)
    Zr = Z + sigma * Ehat / sum(abs(Ehat))
    Zr[train_split_idxs] = Z[train_split_idxs]
    return Zr


def correct_and_smooth(y, yhat, 
                       edge_index, 
                       val_split_idxs=[], 
                       verbose=False, 
                       alpha1=0.8, 
                       alpha2=0.8, 
                       eps1=1e-5, 
                       eps2=1e-5):
    """
    c&s full pipeline 
    """
    train_split_idxs = [ ix for ix in range(len(y)) if ix not in val_split_idxs ]
    S = normalize_adj_matrix(edge_index)
    E = residual_error(y, yhat, val_split_idxs)
    Ehat = smooth_operator(E, S, verbose=verbose, alpha=alpha1, eps=eps1) # correct
    G = autoscale(E, Ehat, yhat, train_split_idxs)
    G[train_split_idxs] = y[train_split_idxs]
    yhat = smooth_operator(G, S, verbose=verbose, alpha=alpha1, eps=eps1) # smooth
    return yhat
