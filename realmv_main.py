import utils
import sys
from model import LogisticRegression
from transform import *
from build_data import build_data
import numpy as np
from sklearn.impute import SimpleImputer
from copy import deepcopy
import pandas as pd
from experiment import run_experiment
import os

data_dir = "data/realmv_datasets_pmnn"

datasets = [d for d in os.listdir(data_dir) if d[0] != "."]

for dataset in datasets:
    for seed in range(5):
        result_dir = "realmv_1218/seed_{}".format(seed)
        X, y = utils.load_data(data_dir, dataset, keepna=True)
        X_train_mv, y_train, X_val_mv, y_val, X_test_mv, y_test = build_data(X, y, random_state=seed)

        print("size:", X_train_mv.shape, "percent_mv", np.isnan(X_train_mv).mean())
        params = {
            "num_epochs": 15000,
            "device": "cpu",
        }

        input_dim = X_train_mv.shape[1]
        output_dim = len(set(y_train))
        placeholder = min(np.nanmin(X_train_mv), np.nanmin(X_val_mv), np.nanmin(X_test_mv)) - 200
        placeholder_imputer = SimpleImputer(strategy='constant', fill_value=placeholder)
        X_train_placeholder = placeholder_imputer.fit_transform(X_train_mv)
        X_val_placeholder = placeholder_imputer.transform(X_val_mv)
        X_test_placeholder = placeholder_imputer.transform(X_test_mv)

        mean_imputer = SimpleImputer(strategy='mean')
        X_train_default = mean_imputer.fit_transform(X_train_mv)
        X_val_default = mean_imputer.transform(X_val_mv)
        X_test_default = mean_imputer.transform(X_test_mv)

        model = LogisticRegression(input_dim, output_dim).to(device=params["device"])
        dirty_mean = np.nanmean(X_train_mv, axis=0)
        transformer = MVImputer(input_dim, dirty_mean.reshape(1, -1), nan=placeholder).to(device=params["device"])
        save_dir = utils.makedir([result_dir, dataset, "Impute"])
        run_experiment(model, transformer, params, X_train_placeholder, y_train, X_val_placeholder, y_val, X_test_placeholder, y_test, save_dir)

        save_dir = utils.makedir([result_dir, dataset, "Mean"])
        model = LogisticRegression(input_dim, output_dim).to(device=params["device"])
        transformer = None
        run_experiment(model, transformer, params, X_train_default, y_train, X_val_default, y_val, X_test_default, y_test, save_dir)
