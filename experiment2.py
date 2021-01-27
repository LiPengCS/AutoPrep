import torch
import utils
from dataloader import get_loader_dict
from train import train_evaluate

def run_experiment(model, transformer, params, X_train, y_train, X_val, y_val, X_test, y_test, save_dir, pretrained_model=None):
    best_logger = None
    best_res = None
    best_val_acc = float("-inf")

    learning_rates = [1e-4, 1e-3, 1e-2, 1e-1]
    weight_decays = [0]

    for lr in learning_rates:
        for wd in weight_decays:
            print("learning rate", lr, ", weight decay", wd)

            params["learning_rate"] = lr
            params["weight_decay"] = wd

            torch.manual_seed(1)
            if params["device"] == "cuda":
                torch.cuda.manual_seed(1)
            loader_dict = get_loader_dict(X_train, y_train, X_val, y_val, X_test, y_test, params["batch_size"])

            if pretrained_model is not None:
                model.load_state_dict(deepcopy(pretrained_model.state_dict()))

            loss_fn = torch.nn.CrossEntropyLoss()

            if transformer is not None:
                model_optimizer = torch.optim.SGD(list(model.parameters()) + list(transformer.parameters()), lr=params["learning_rate"], weight_decay=params["weight_decay"])
            else:
                model_optimizer = torch.optim.SGD(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])

            # model_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer)

            model_scheduler = None
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
