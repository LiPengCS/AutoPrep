from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np

def numpy_to_loader(X, y, batch_size):
    X = torch.Tensor(X)
    y = torch.tensor(y)
    # dataset = TensorDataset(X, y) 
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader = [(X, y)]
    return dataloader

def get_loader_dict(X_train, y_train, X_val, y_val, X_test, y_test, batch_size):
    train_loader = numpy_to_loader(X_train, y_train, batch_size)
    val_loader = numpy_to_loader(X_val, y_val, batch_size)
    test_loader = numpy_to_loader(X_test, y_test, batch_size)
    loader_dict = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }
    return loader_dict

