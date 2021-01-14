import utils
from model import LogisticRegression
from dataloader import get_loader_dict
from train import train_evaluate
import torch
from transform import *
from build_data import build_data
import sys
from copy import deepcopy
import argparse

def run(transformer, save_dir, pretrained_model=None):
    best_logger = None
    best_res = None
    best_val_acc = float("-inf")

    learning_rates = [1e-4, 1e-3, 1e-2, 1e-1]
    weight_decays = [0]

    for lr in learning_rates:
        for wd in weight_decays:
            print(lr, wd)
            input_dim = X_train.shape[1]
            output_dim = len(set(y_train))
            assert output_dim == 2

            params["learning_rate"] = lr
            params["weight_decay"] = wd

            torch.manual_seed(1)
            if params["device"] == "cuda":
                torch.cuda.manual_seed(1)
            loader_dict = get_loader_dict(X_train, y_train, X_val, y_val, X_test, y_test, params["batch_size"])

            model = LogisticRegression(input_dim, output_dim).to(device=params["device"])
            if pretrained_model is not None:
                model.load_state_dict(deepcopy(pretrained_model.state_dict()))

            loss_fn = torch.nn.CrossEntropyLoss()
            model_optimizer = torch.optim.SGD(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
            model_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer)

            if transformer is not None:
                transformer_optimizer = torch.optim.SGD(transformer.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
                transformer_schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(transformer_optimizer)
            else:
                transformer_optimizer = None
                transformer_schedular = None
            
            res, logger = train_evaluate(model, model_optimizer, model_scheduler, loader_dict, loss_fn, params,
                                         transformer=transformer, transformer_optimizer=transformer_optimizer, 
                                         transformer_schedular=transformer_schedular, warm_start=False)

            if res["val_acc"] > best_val_acc:
                best_res = res
                best_val_acc = res["val_acc"]
                best_logger = logger
                best_res["learning_rate"] = lr
                best_res["weight_decay"] = wd
                best_model = model

    utils.save_result(best_res, best_logger, save_dir)
    return best_model


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=None)
parser.add_argument('--result_dir', default='./result')
parser.add_argument('--warm_start', action='store_true', default=False)
args = parser.parse_args()

data_dir = "data"
X, y = utils.load_data(data_dir, args.dataset)
X_train, y_train, X_val, y_val, X_test, y_test = build_data(X, y)

params = {
    "num_epochs": 20000,
    "batch_size": 10000,
    "device": "cpu",
    "warm_start_epochs": 2000
}

input_dim = X_train.shape[1]

save_dir = utils.makedir([args.result_dir, args.dataset, "None"])
baseline_model = run(None, save_dir)

save_dir = utils.makedir([args.result_dir, args.dataset, "Normalize"])
transformer = StandardScaler(input_dim).to(device=params["device"])
run(transformer, save_dir)

save_dir = utils.makedir([args.result_dir, args.dataset, "Normalize_warmstart"])
transformer = StandardScaler(input_dim).to(device=params["device"])
run(transformer, save_dir, pretrained_model=baseline_model)





