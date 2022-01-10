"""
The first step is to acquire a base classifier model that can output a probability distribution over the classes. We train a shallow MLP in PyTorch:
"""
import torch

class BasePredictor(torch.nn.Module):
    """
    A simple MLP class to serve as the base predictor
    """
    def __init__(self, n_hidden_layers=1, in_size=128, hidden_size=64, out_size=1):
        super(BasePredictor, self).__init__()
        if n_hidden_layers == 0:
            self.net = torch.nn.Linear(in_size, out_size)
        else:
            net  = [torch.nn.Linear(in_size, hidden_size), torch.nn.ReLU()]
            net += [torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU()] * (n_hidden_layers - 1)
            net += [torch.nn.Linear(hidden_size, out_size)]
            self.net = torch.nn.Sequential(*net)

    def forward(self, X):
        out = self.net(X)
        return out.squeeze()
