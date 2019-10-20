import torch
import torch.nn as nn
import torch.nn.functional as F
import hyp

class QNetwork(nn.Module):
    """A four layered fully-connected network to approximate the Q value."""
    
    def __init__(self):
        super(QNetwork, self).__init__()
        self.linear1 = nn.Linear(hyp.fingerprint_length, hyp.dense_layers[0])
        self.linear2 = nn.Linear(hyp.dense_layers[0], hyp.dense_layers[1])
        self.linear3 = nn.Linear(hyp.dense_layers[1], hyp.dense_layers[2])
        self.linear4 = nn.Linear(hyp.dense_layers[2], hyp.dense_layers[3])
        self.out = nn.Linear(hyp.dense_layers[3], 1)

    def forward(self, x):
        activation = getattr(F, hyp.activation)
        x = activation(self.linear1(x))
        x = activation(self.linear2(x))
        x = activation(self.linear3(x))
        x = activation(self.linear4(x))
        return self.out(x)
