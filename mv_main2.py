from build_data import build_data
from sklearn.impute import SimpleImputer
from experiment2 import run_experiment
import utils
import numpy as np
from transform2 import MVGridImputeLayer
import torch
from model import LogisticRegression
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=None)
parser.add_argument('--data_dir', default="data/realmv_datasets_pmnn")
parser.add_argument('--result_dir', default="result")
args = parser.parse_args()

if args.dataset is not None:
    datasets = [args.dataset]
else:
    # run all datasets if dataset is not specified
    datasets = [d for d in os.listdir(args.data_dir) if d[0] != "."]

for dataset in datasets:
    print(dataset)

    # load data
    X, y = utils.load_data(args.data_dir, dataset, keepna=True)

    # split data to train/val/test and preprocess data
    X_train_mv, y_train, X_val_mv, y_val, X_test_mv, y_test = build_data(X, y)

    # set mv to min-200
    placeholder = min(np.nanmin(X_train_mv), np.nanmin(X_val_mv), np.nanmin(X_test_mv)) - 200
    placeholder_imputer = SimpleImputer(strategy='constant', fill_value=placeholder)
    X_train = placeholder_imputer.fit_transform(X_train_mv)
    X_val = placeholder_imputer.transform(X_val_mv)
    X_test = placeholder_imputer.transform(X_test_mv)
    X_train = torch.Tensor(X_train)
    X_val = torch.Tensor(X_val)
    X_test = torch.Tensor(X_test)

    params = {
        "num_epochs": 10000,
        "batch_size": 15000, # I use batch gradient descent. batch size is not used actually.
        "device": "cpu",
    }

    input_dim = X_train.shape[1]
    output_dim = len(set(y_train))

    ## our approach
    # initialize imputer
    mv_imputer = MVGridImputeLayer(X_train_mv, placeholder, n_grid=5)
    model = LogisticRegression(input_dim, output_dim).to(device=params["device"])
    save_dir = utils.makedir([args.result_dir, dataset, "imputation"])
    run_experiment(model, mv_imputer, params, X_train, y_train, X_val, y_val, X_test, y_test, save_dir)

    ## baseline: mean imputation
    mean_imputer = SimpleImputer(strategy='mean')
    X_train_default = mean_imputer.fit_transform(X_train_mv)
    X_val_default = mean_imputer.transform(X_val_mv)
    X_test_default = mean_imputer.transform(X_test_mv)
    save_dir = utils.makedir([args.result_dir, dataset, "Mean"])
    model = LogisticRegression(input_dim, output_dim).to(device=params["device"])
    transformer = None
    run_experiment(model, transformer, params, X_train_default, y_train, X_val_default, y_val, X_test_default, y_test, save_dir)