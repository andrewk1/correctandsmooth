import torch

def edge_index_to_sparse(edge_index):
    return torch.sparse_coo_tensor(edge_index, values=torch.ones(edge_index.shape[-1]))
