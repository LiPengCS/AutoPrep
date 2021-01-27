import numpy as np
import torch
import torch.nn as nn
import utils

class MVGridImputeLayer(nn.Module):
    def __init__(self, X_train_mv, nan, n_grid=5):
        super(MVGridImputeLayer, self).__init__()
        self.nan = nan + 100
        self.n_grid = n_grid
        max_x = np.nanmax(X_train_mv, axis=0)
        min_x = np.nanmin(X_train_mv, axis=0)
        self.imp_values = []
        for s, l in zip(min_x, max_x):
            imp = np.linspace(s, l, num=n_grid)
            self.imp_values.append(imp)
        self.imp_values = torch.Tensor(self.imp_values).T
        weights = torch.ones((n_grid, X_train_mv.shape[1])) / n_grid
        self.sigmoid = nn.Sigmoid()
        self.register_parameter(name='weights', param=torch.nn.Parameter(weights))

    def forward(self, x, mode):
        m = x.shape[1]
        X_imp = []
        for i in range(self.n_grid):
            imp = self.sigmoid(100 * (x - self.nan)) * (x - self.imp_values[i]) + self.imp_values[i]
            X_imp.append(imp.unsqueeze(1))
        X_imp = torch.cat(X_imp, dim=1)

        X = []
        for i in range(m):
            X_i = X_imp[:, :, i].mm(self.weights[:, i:i+1])
            X.append(X_i)
        X = torch.cat(X, dim=1)
        return X