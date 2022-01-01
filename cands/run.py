from cands import correct_and_smooth

"""
Standard utility functionality to load and generate dataset splits
"""
import numpy as np
from torch_geometric.datasets import Twitch, LastFMAsia

graph = LastFMAsia("./data")  # Twitch("./data", "EN")
X = graph.data.x
edge_index = graph.data.edge_index
y = graph.data.y.squeeze()

NUM_EDGE_KEEP = 0.3

kept_edges = np.random.choice(edge_index.shape[1], int(edge_index.shape[1] * NUM_EDGE_KEEP))
edge_index = edge_index[:, kept_edges]
kept_nodes = list(set(edge_index.flatten().tolist()))

# We use a 0.8 / 0.1 / 0.1 train/val/test split
# For the last section of the blog post, we attempt an experiment with much
# smaller data regime (0.5, 0.25, 0.25)
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