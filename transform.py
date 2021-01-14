import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class StandardScaler(nn.Module):
    def __init__(self, dim):
        super(StandardScaler, self).__init__()
        mean = Variable(torch.zeros(1, dim), requires_grad=True)
        std = Variable(torch.zeros(1, dim), requires_grad=True)
        self.register_parameter(name='mean', param=torch.nn.Parameter(mean))
        self.register_parameter(name='std', param=torch.nn.Parameter(std))

    def forward(self, x, mode=None):
        x = (x - x.mean(axis=0, keepdim=True)) / x.std(axis=0, keepdim=True)
        outputs = x * (self.std + 1) + self.mean
        return outputs

class NoneTransformer(nn.Module):
    def __init__(self):
        super(NoneTransformer, self).__init__()
        pass

    def forward(self, x, mode=None):
        return x

class MVImputer(nn.Module):
    def __init__(self, dim, mean, nan=-100):
        super(MVImputer, self).__init__()
        imputation = Variable(torch.Tensor(mean), requires_grad=True)
        self.nan = nan + 100
        self.register_parameter(name='imputation', param=torch.nn.Parameter(imputation))

    def forward(self, x, mode=None):
        outputs = 1 / (1 + torch.exp(-100 * (x - self.nan))) * (x - self.imputation) + self.imputation
        return outputs

class DropoutOld(nn.Module):
    def __init__(self, p=0.1):
        super(DropoutOld, self).__init__()
        drop_prob = Variable(torch.Tensor([p]), requires_grad=True)
        self.register_parameter(name='drop_prob', param=torch.nn.Parameter(drop_prob))

    def forward(self, x, mode="train"):
        if mode == "train":
            R = torch.rand(*x.shape)
            outputs = x * (1 / (1 + torch.exp(-100 * (R - self.drop_prob))))
            return outputs
        else:
            return x

class DropFeature(nn.Module):
    """docstring for DropFeature"""
    def __init__(self, m, init_p=0):
        super(DropFeature, self).__init__()
        drop_prob = Variable(torch.ones((1, m)) * init_p, requires_grad=True)
        self.register_parameter(name='drop_prob', param=torch.nn.Parameter(drop_prob))

    def forward(self, x, mode="train"):
        outputs = x * (1 / (1 + torch.exp(-100 * (self.drop_prob))))
        return outputs

class Dropout(nn.Module):
    """docstring for DropFeature"""
    def __init__(self, X_shape, init_p=0):
        super(Dropout, self).__init__()
        drop_prob = Variable(torch.ones(*X_shape) * init_p, requires_grad=True)
        self.register_parameter(name='drop_prob', param=torch.nn.Parameter(drop_prob))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mode="train"):
        if mode == "train":
            outputs = x * 2 * self.sigmoid(-200 * self.drop_prob)
            return outputs
        else:
            return x

    def reg(self, x):
        outputs = x * 2 * self.sigmoid(-200 * self.drop_prob)
        delta = torch.sum((x - outputs)**2)
        return delta

class Discretization(nn.Module):
    def __init__(self, X_train, init_n_split=10):
        super(Discretization, self).__init__()
        self.x_max = torch.Tensor(X_train.max(axis=0).reshape(1, -1))
        self.x_min = torch.Tensor(X_train.min(axis=0).reshape(1, -1))
        N, m = X_train.shape
        split = Variable(torch.ones((1, m)) * init_n_split, requires_grad=True)
        # interval_length = (self.x_max - self.x_min) / init_n_split
        # split = Variable(torch.Tensor(interval_length), requires_grad=True)
        self.register_parameter(name='split', param=torch.nn.Parameter(split))
        self.sigmoid = nn.Sigmoid()

    def floor(self, a, max_n=20):
        result = torch.zeros(*a.shape)
        for i in range(1, max_n):
            result += self.sigmoid(100 * (a-i))
        return result

    def forward(self, x, mode="train"):
        interval_length = (self.x_max - self.x_min) / self.split
        outputs = self.floor((x - self.x_min) / interval_length)
        return outputs

class DiscretizationIdentity(nn.Module):
    def __init__(self, X_train, init_n_split=10):
        super(Discretization, self).__init__()
        self.x_max = torch.Tensor(X_train.max(axis=0).reshape(1, -1))
        self.x_min = torch.Tensor(X_train.min(axis=0).reshape(1, -1))
        N, m = X_train.shape
        split = Variable(torch.ones((1, m)) * init_n_split, requires_grad=True)
        # interval_length = (self.x_max - self.x_min) / init_n_split
        # split = Variable(torch.Tensor(interval_length), requires_grad=True)
        self.register_parameter(name='split', param=torch.nn.Parameter(split))
        self.sigmoid = nn.Sigmoid()

    def floor(self, a, max_n=20):
        result = torch.zeros(*a.shape)
        for i in range(1, max_n):
            result += self.sigmoid(100 * (a-i))
        return result

    def forward(self, x, mode="train"):
        interval_length = (self.x_max - self.x_min) / self.split
        outputs = self.floor((x - self.x_min) / interval_length)
        return outputs



