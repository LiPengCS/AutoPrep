import numpy as np
import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

class NN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NN, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, 5)
        self.linear2 = torch.nn.Linear(5, 5)
        self.linear3 = torch.nn.Linear(5, 5)
        self.linear4 = torch.nn.Linear(5, 5)
        self.linear5 = torch.nn.Linear(5, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.relu(x)
        outputs = self.linear5(x)
        return outputs