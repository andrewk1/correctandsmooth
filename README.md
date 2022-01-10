### simple correct&smooth (cands)

Simple implementation of [correct and smooth](https://arxiv.org/abs/2010.13993) (C&S) in pytorch.
`cands` is intended to be very quick for practictioners to pick up and run as a postprocessing step on your current classifiers.
While lots of data contains some form of graph structure, it is rarely used in practice due to the challenges of setting up
GNNs or extracting manual graph features. 

#### What tasks work well with C&S
C&S works for transductive node classification tasks, which means that the graph itself remains static and accessible during
both traing and validation. After you have generated predictions, C&S will "smooth" them using the neighboring
training node features and labels. We've seen up to [20% increase in validation performance](https://colab.research.google.com/drive/1RgRm-xKnKObYsHGbXB9TRYUKriAj3yV5?usp=sharing) when running C&S over neural network-based predictions on the features alone.

#### Installation
Install directly:
```
pip install https://github.com/andrewk1/correctandsmooth/archive/master.zip
```

Install from source
```
git clone https://github.com/andrewk1/correctandsmooth.git && cd correctandsmooth
pip install .
```

`cands` has been tested on `torch>=1.9`.

#### Usage
C&S is a single postprocessing step applied to your predictions, we expose it via a function
```
from cands import correct_and_smooth

yhat_cands = correct_and_smooth(y, yhat, edge_index, val_split_idxs)
```
Currently, C&S needs four arguments:\

`y`: Your current labels in matrix form `(N, D)` where `N` is your dataset length and `D` is the number of classes.\
`yhat`: Predictions of shape `(N, D)`\
`edge_index`: Tensor of shape `(2, E)` where each column contains indices `0 <= i < N` of nodes with shared edges. `E` is the total edge count\
`val_split_idxs`: Indices of validation or unlabeled nodes (nodes not in the training set)\

