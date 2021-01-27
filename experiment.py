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
            print(lr, wd)

            params["learning_rate"] = lr
            params["weight_decay"] = wd

            torch.manual_seed(1)
            if params["device"] == "cuda":
                torch.cuda.manual_seed(1)
            loader_dict = get_loader_dict(X_train, y_train, X_val, y_val, X_test, y_test, params["batch_size"])

            if pretrained_model is not None:
                model.load_state_dict(deepcopy(pretrained_model.state_dict()))

            loss_fn = torch.nn.CrossEntropyLoss()
            model_optimizer = torch.optim.SGD(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
            # model_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer)
            model_scheduler = None

            if transformer is not None:
                transformer_optimizer = torch.optim.SGD(transformer.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
                # transformer_schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(transformer_optimizer)
                transformer_schedular = None
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

def run_dropout_experiment(model, transformer, params, X_train, y_train, X_val, y_val, X_test, y_test, save_dir, pretrained_model=None):
    best_logger = None
    best_res = None
    best_val_acc = float("-inf")

    learning_rates = [1e-3, 1e-2, 1e-1]
    regularization = [1e-3, 1e-2, 1e-1, 0]

    # weight_decays = [0, 1e-1, 1e-5]
    # learning_rates = [1e-3]
    weight_decays = [0]

    for lr in learning_rates:
        for wd in weight_decays:
            for reg in regularization:
                print(lr, reg)
                params["learning_rate"] = lr
                params["weight_decay"] = wd
                params["reg"] = reg

                torch.manual_seed(1)
                if params["device"] == "cuda":
                    torch.cuda.manual_seed(1)
                loader_dict = get_loader_dict(X_train, y_train, X_val, y_val, X_test, y_test, params["batch_size"])

                if pretrained_model is not None:
                    model.load_state_dict(deepcopy(pretrained_model.state_dict()))

                loss_fn = torch.nn.CrossEntropyLoss()
                model_optimizer = torch.optim.SGD(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
                # model_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer)
                model_scheduler = None

                if transformer is not None:
                    transformer_optimizer = torch.optim.SGD(transformer.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
                    # transformer_schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(transformer_optimizer)
                    transformer_schedular = None
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
                    best_res["reg"] = reg
                    best_model = model

    utils.save_result(best_res, best_logger, save_dir)
    return best_model