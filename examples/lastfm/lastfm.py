"""
Using cands to implement and evaluate C&S on AsiaFM Dataset

Additional dependencies:
- torch-geometric
"""

import numpy as np
import torch; torch.manual_seed(42);
import torch.nn.functional as F

from cands import correct_and_smooth
from torch_geometric.datasets import Twitch, LastFMAsia
from tqdm import tqdm

from model import BasePredictor

graph = LastFMAsia("./data")  # Twitch("./data", "EN")
X = graph.data.x
edge_index = graph.data.edge_index
y = graph.data.y.squeeze()

NUM_EDGE_KEEP = 0.3

kept_edges = np.random.choice(edge_index.shape[1], int(edge_index.shape[1] * NUM_EDGE_KEEP))
edge_index = edge_index[:, kept_edges]
kept_nodes = list(set(edge_index.flatten().tolist()))

# We use a 0.5 / 0.25 / 0.25 train/val/test split
SPLIT_FRACTIONS = (0.8,  0.1, 0.1)
splits_sizes = (int(SPLIT_FRACTIONS[0] * len(X[kept_nodes])), 
                int(SPLIT_FRACTIONS[1] * len(X[kept_nodes])), 
                len(X[kept_nodes]) - int(SPLIT_FRACTIONS[0] * len(X[kept_nodes]))- int(SPLIT_FRACTIONS[1] * len(X[kept_nodes])))
train_split, val_split, test_split = splits = torch.utils.data.random_split(X[kept_nodes], splits_sizes)
(X_train, y_train), (X_val, y_val), (X_test, y_test) = [(X[split.indices], y[split.indices]) for split in splits] 

num_labels = int(max(y) + 1)
print(f"Dataset: { X_train.shape[0] } training, { X_val.shape[0] } val, { X_test.shape[0] } test samples with { X.shape[1] } dim embeddings")
print(f"{ edge_index.shape[1] } total followerships (edges)")
print(f"{ num_labels } total classes")

net = BasePredictor(in_size=X.shape[1], n_hidden_layers=1, out_size=num_labels)
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())

def train(X, y):
    optimizer.zero_grad()
    yhat = net(X)
    l = loss(yhat, y)
    l.backward()
    optimizer.step()
    return l

NUM_EPOCHS = 500

pbar = tqdm(range(NUM_EPOCHS))
for ep in pbar:
    l = train(X_train, y_train)
    pred = torch.argmax(net(X_val), -1)
    pbar.set_postfix({'loss': float(l), "val_acc": float(torch.sum(pred == y_val) / len(pred))})

yhat = torch.softmax(net(X), -1)
val_split_idxs = val_split.indices + test_split.indices
yhat_cands = correct_and_smooth(y, yhat, edge_index, val_split_idxs)

def hyperparameter_sweep(model, X, y, alpha1s, alpha2s):
    """
    We test val accuracy over a grid search of alpha1 and alpha2 and return
    the results as a list of (val_acc, (alpha1, alpha2)) for each run.
    """
    results = []
    Z = torch.sigmoid(model(X))
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


alpha1s, alpha2s = np.linspace(0.1, 0.9, 5), np.linspace(0.1, 0.9, 5)
sweep = sorted(hyperparameter_sweep(net, X, y, alpha1s, alpha2s))
print(f"Max val acc: { sweep[-1][0] } with hparams: { sweep[-1][1] }")

"""Here below we summarize our model performance results. With each step, we see nearly 10% jump in accuracy! Clearly, there are huge gains to be had in including the graph structure in this particular predictive task. We also see the importance of the two alpha variables to the performance of the smoothing steps - run a hyperparameter sweep if you choose to implement this method!"""

yhat_mlp = torch.argmax(net(X), -1)
print(f"Val  accuracy MLP: { torch.mean((yhat_mlp[val_split.indices] == y[val_split.indices]).type(torch.float32)) }")
print(f"Test accuracy MLP: { torch.mean((yhat_mlp[test_split.indices] == y[test_split.indices]).type(torch.float32)) }\n")

yhat_correct = torch.argmax(G, -1)
print(f"Val  accuracy Correct: { torch.mean((yhat_correct[val_split.indices] == y[val_split.indices]).type(torch.float32)) }")
print(f"Test accuracy Correct: { torch.mean((yhat_correct[test_split.indices] == y[test_split.indices]).type(torch.float32)) }\n")

yhat_cs = torch.argmax(correct_and_smooth(E, Z, y), -1)
print(f"Val  accuracy Correct&Smooth: { torch.mean((yhat_cs[val_split.indices] == y[val_split.indices]).type(torch.float32)) }")
print(f"Test accuracy Correct&Smooth: { torch.mean((yhat_cs[test_split.indices] == y[test_split.indices]).type(torch.float32)) }\n")

yhat_cs_sweep = torch.argmax(sorted(sweep)[-1][2], -1)
print(f"Val  accuracy Correct&Smooth Sweep: { torch.mean((yhat_cs_sweep[val_split.indices] == y[val_split.indices]).type(torch.float32)) }")
print(f"Test accuracy Correct&Smooth Sweep: { torch.mean((yhat_cs_sweep[test_split.indices] == y[test_split.indices]).type(torch.float32)) }\n")

"""#### Explaining C&S
A 20% increase in accuracy deserves some scrutiny - why does C&S perform so well here? Dense graphs lend themselves really well to classical smoothing approaches.
"""

Y_simple = Y.clone().type(torch.float)
Y_simple[val_split.indices + test_split.indices] = 0

Yhat_simple = simple_smooth(Y_simple, verbose=False)
yhat_simple_correct = torch.argmax(Yhat_simple, -1)
print(f"Val  accuracy Correct: { torch.mean((yhat_simple_correct[val_split.indices] == y[val_split.indices]).type(torch.float32)) }")
print(f"Test accuracy Correct: { torch.mean((yhat_simple_correct[test_split.indices] == y[test_split.indices]).type(torch.float32)) }\n")

