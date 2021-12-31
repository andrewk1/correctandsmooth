"""
Correct&Smooth Implementation
"""
import torch
import numpy as np

from tqdm import tqdm


def to_dense_adj(edge_index):
    """"""
    raise NotImplementedError()

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


def from_numpy(x):
    if isinstance(x, (np.ndarray, np.generic)):
        return torch.from_numpy(x)
    assert (torch.is_tensor(x)), f"Recieved arg with type {type(x)} but need torch.tensor"
    return x

class CorrectAndSmooth:
    """
    """
    def __init__(self, data, target, preds, test_split, edge_index, val_split=[]) -> None:
        self.data = from_numpy(data)
        self.target = from_numpy(target)
        self.preds = from_numpy(preds)

        self.test_split, self.val_split = test_split, val_split
        self.train_split = [x for x in range(len(data)) if (x not in self.test_split and x not in self.val_split)]

        self.edge_index = edge_index
        self.S = normalize_adj_matrix(self.edge_index)

    def residual_error(self, Z):
        """
        Form E, residual error matrix Z - L for training data"
        """
        E = Z - self.target
        E[self.val_split + self.test_split] = 0
        return E

    def correct(self, E, alpha1 = 0.8, eps = 1e-5, verbose=True):
        """

        E^(t+1) = (1-alpha1)E + alpha * S @ E^(t) -> Ehat
        """
        return smooth_operator(E, self.S, alpha1, eps, verbose)

    def autoscale(self, E, Ehat, Z):
        """
        sigma = sum of absolute value of E for each training sample / num training samples
        """
        sigma = float(sum(torch.norm(E[self.train_split], p=1, dim=-1))) / len(self.train_split)
        Zr = Z + sigma * Ehat / sum(abs(Ehat))
        Zr[self.train_split.indices] = Z[self.train_split.indices]
        return Zr

    def smooth(self, G, alpha2=0.8, eps=1e-5, verbose=True):
        """"""
        return smooth_operator(G, self.S, alpha2, eps, verbose)

    def correct_and_smooth_sweep(self, E, Z, y, alpha1=0.4, alpha2=0.4):
        """
        Full pipeline for C&S
        """
        Ehat = self.correct(E, alpha1=alpha1, verbose=False, eps=1e-4)
        G = self.autoscale(E, Ehat, Z)
        G[self.train_split.indices] = Y[self.train_split.indices].type(torch.float32)
        yhat = self.smooth(G, alpha2=alpha2, verbose=False, eps=1e-4)
        return yhat

    def correct_and_smooth(self):
        E = 

    def simple_smooth(Y_simple, alpha_simple = 0.8, eps = 1e-5, verbose=True):
        if verbose:
            pbar = tqdm(total=float('inf'))

        Yhat_simple = Y_simple
        diff = eps
        itr = 0
        while diff >= eps:
            # This is the iterative update step
            Yhat_t = (1 - alpha_simple) * Y_simple + alpha_simple * (S @ Yhat_simple)
            diff = float(torch.norm(Yhat_simple - Yhat_t))
            Yhat_simple = Yhat_t

            if verbose:
                pbar.update(1)
                pbar.set_postfix({ 'diff': diff })

        return Yhat_simple


    def hyperparameter_sweep(model, X, y, alpha1s, alpha2s):
        """
        We test val accuracy over a grid search of alpha1 and alpha2 and return
        the results as a list of (val_acc, (alpha1, alpha2)) for each run.
        """
        results = []
        Z = preds 
        E = residual_error(Z)
        with tqdm(total=len(alpha1s) * len(alpha2s)) as pbar:
            for alpha1 in alpha1s:
                for alpha2 in alpha2s:
                    yhat = correct_and_smooth(E, Z, y, alpha1, alpha2)
                    pred = torch.argmax(yhat, -1)
                    val_acc = torch.mean((pred[val_split.indices] == y[val_split.indices]).type(torch.float32))
                    results.append([float(val_acc), (alpha1, alpha2), yhat])
                    pbar.update(1)
        return results