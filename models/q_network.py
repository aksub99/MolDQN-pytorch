import torch
import torch.nn as nn
import torch.nn.functional as F
import hyp

class QNetwork(nn.Module):
    """A four layered fully-connected network to approximate the Q values."""
    
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.linear1 = nn.Linear(state_dim, hyp.dense_layers[0])
        self.linear2 = nn.Linear(hyp.dense_layers[0], hyp.dense_layers[1])
        self.linear3 = nn.Linear(hyp.dense_layers[1], hyp.dense_layers[2])
        self.linear4 = nn.Linear(hyp.dense_layers[2], hyp.dense_layers[3])
        self.out = nn.Linear(hyp.dense_layers[3], action_dim)

    def forward(self, state):
        activation = getattr(F, hyp.activation)
        q_vals = activation(self.linear1(state))
        q_vals = activation(self.linear2(q_vals))
        q_vals = activation(self.linear3(q_vals))
        q_vals = activation(self.linear4(q_vals))
        q_vals = self.out(q_vals)
        return q_vals
