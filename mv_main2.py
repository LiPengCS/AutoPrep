from build_data import build_data
from sklearn.impute import SimpleImputer
from experiment2 import run_experiment
import utils
import numpy as np
from transform_new import MVGridImputeLayer
import torch
from model import LogisticRegression
import argparse
import os

data_dir = "data/realmv_datasets_pmnn"
result_dir = "result0122_2"
datasets = [d for d in os.listdir(data_dir) if d[0] != "."]

for dataset in datasets:
    print(dataset)
    X, y = utils.load_data(data_dir, dataset, keepna=True)
    X_train_mv, y_train, X_val_mv, y_val, X_test_mv, y_test = build_data(X, y)
    placeholder = min(np.nanmin(X_train_mv), np.nanmin(X_val_mv), np.nanmin(X_test_mv)) - 200
    placeholder_imputer = SimpleImputer(strategy='constant', fill_value=placeholder)
    mv_imputer = MVGridImputeLayer(X_train_mv, placeholder, n_grid=2)

    X_train = placeholder_imputer.fit_transform(X_train_mv)
    X_val = placeholder_imputer.transform(X_val_mv)
    X_test = placeholder_imputer.transform(X_test_mv)

    X_train = torch.Tensor(X_train)
    X_val = torch.Tensor(X_val)
    X_test = torch.Tensor(X_test)

    params = {
        "num_epochs": 20000,
        "batch_size": 15000,
        "device": "cpu",
    }

    input_dim = X_train.shape[1]
    output_dim = len(set(y_train))

    mean_imputer = SimpleImputer(strategy='mean')
    X_train_default = mean_imputer.fit_transform(X_train_mv)
    X_val_default = mean_imputer.transform(X_val_mv)
    X_test_default = mean_imputer.transform(X_test_mv)
    save_dir = utils.makedir([result_dir, dataset, "Mean"])
    model = LogisticRegression(input_dim, output_dim).to(device=params["device"])
    transformer = None
    run_experiment(model, transformer, params, X_train_default, y_train, X_val_default, y_val, X_test_default, y_test, save_dir)

    model = LogisticRegression(input_dim, output_dim).to(device=params["device"])
    save_dir = utils.makedir([result_dir, dataset, "imputation"])
    run_experiment(model, mv_imputer, params, X_train, y_train, X_val, y_val, X_test, y_test, save_dir)


