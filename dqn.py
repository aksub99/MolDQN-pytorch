import torch
import torch.nn as nn
import torch.nn.functional as F


class MolDQN(nn.Module):
    def __init__(self, input_length, output_length):
        super(MolDQN, self).__init__()

        self.linear_1 = nn.Linear(input_length, 1024)
        self.linear_2 = nn.Linear(1024, 512)
        self.linear_3 = nn.Linear(512, 128)
        self.linear_4 = nn.Linear(128, 32)
        self.linear_5 = nn.Linear(32, output_length)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.linear_1(x))
        x = self.activation(self.linear_2(x))
        x = self.activation(self.linear_3(x))
        x = self.activation(self.linear_4(x))
        x = self.linear_5(x)

        return x
