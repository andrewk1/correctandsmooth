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

yhat_mlp = torch.argmax(net(X), -1)
print(f"Val  accuracy MLP: { torch.mean((yhat_mlp[val_split.indices] == y[val_split.indices]).type(torch.float32)) }")
print(f"Test accuracy MLP: { torch.mean((yhat_mlp[test_split.indices] == y[test_split.indices]).type(torch.float32)) }\n")

yhat_cands = torch.argmax(yhat_cands, -1)
print(f"Val  accuracy CandS: { torch.mean((yhat_cands[val_split.indices] == y[val_split.indices]).type(torch.float32)) }")
print(f"Test accuracy CandS: { torch.mean((yhat_cands[test_split.indices] == y[test_split.indices]).type(torch.float32)) }\n")

