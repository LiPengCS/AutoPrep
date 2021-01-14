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

params = {
    "num_epochs": 10000,
    "device": "cpu",
}

data_dir = "data/normalization_datasets"
result_dir = "dropout_0108"
dataset = sys.argv[1]
X, y = utils.load_data(data_dir, dataset, keepna=True)
X_train, y_train, X_val, y_val, X_test, y_test = build_data(X, y)

input_dim = X_train.shape[1]
output_dim = len(set(y_train))
model = LogisticRegression(input_dim, output_dim).to(device=params["device"])
transformer = Dropout(X_train.shape)
save_dir = utils.makedir([result_dir, dataset, "Dropout"])
run_experiment(model, transformer, params, X_train, y_train, X_val, y_val, X_test, y_test, save_dir)

save_dir = utils.makedir([result_dir, dataset, "Baseline"])
model = LogisticRegression(input_dim, output_dim).to(device=params["device"])
transformer = None
run_experiment(model, transformer, params, X_train, y_train, X_val, y_val, X_test, y_test, save_dir)
