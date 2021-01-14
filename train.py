import torch
import torch.nn as nn
# from tqdm import tqdm
import utils
from collections import defaultdict

# training function 
def train(model, model_optimizer, loader, loss_fn, 
          params, transformer=None, transformer_optimizer=None):
    epoch_loss = 0
    epoch_correct = 0
    n_examples = 0
    model.train()
    
    #with tqdm(total=len(loader)) as t:
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(params["device"])
        y_batch = y_batch.to(params["device"])

        if transformer is not None:
            output_batch = model(transformer(X_batch, mode="train"))
        else:
            output_batch = model(X_batch)
        
        if "reg" in params and transformer is not None:
            l1 =  loss_fn(output_batch, y_batch)
            l2 = params["reg"] * transformer.reg(X_batch)
            loss = l1 + l2
            # print(l1, l2)
        else:
            loss = loss_fn(output_batch, y_batch)
        
        _, preds = torch.max(output_batch, 1)
        correct = (preds == y_batch).sum()

        if transformer_optimizer is not None:
            transformer_optimizer.zero_grad()
            model_optimizer.zero_grad()
            loss.backward()

            transformer_optimizer.step()
            model_optimizer.step()

        else:
            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()
        
        epoch_loss += loss.item()
        epoch_correct += correct.item()
        n_examples += len(y_batch)

    avg_loss = epoch_loss / len(loader)
    avg_acc = epoch_correct / n_examples
    return avg_loss, avg_acc

def evaluate(model, loader, loss_fn, params, transformer=None):
    epoch_loss = 0
    epoch_correct = 0
    n_examples = 0
    model.eval()
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(params["device"])
            y_batch = y_batch.to(params["device"])

            if transformer is not None:
                output_batch = model(transformer(X_batch, mode="test"))
            else:
                output_batch = model(X_batch)

            loss = loss_fn(output_batch, y_batch)

            _, preds = torch.max(output_batch, 1)
            correct = (preds == y_batch).sum()
            
            epoch_loss += loss.item()
            n_examples += len(y_batch)
            epoch_correct += correct.item()

    avg_loss = epoch_loss / len(loader)
    avg_acc = epoch_correct / n_examples
    return avg_loss, avg_acc

def logging(logger, tr_loss, tr_acc, val_loss, val_acc, test_loss, test_acc):
    logger["train_losses"].append(tr_loss)
    logger["val_losses"].append(val_loss)
    logger["test_losses"].append(test_loss)
    logger["train_accs"].append(tr_acc)
    logger["val_accs"].append(val_acc)
    logger["test_accs"].append(test_acc)

def train_evaluate(model, model_optimizer, model_scheduler, loader_dict, 
                   loss_fn, params, transformer=None, transformer_optimizer=None,
                   transformer_schedular=None, warm_start=False):
    start_epoch = 0
    best_val_acc = 0
    logger = {"train_losses":[], "val_losses":[], "train_accs":[], 
              "val_accs":[], "test_losses":[], "test_accs": [], 
              "transformer_params":defaultdict(list)}

    if transformer is not None and warm_start:
        for e in range(params["warm_start_epochs"]):
            tr_loss, tr_acc = train(model, model_optimizer, loader_dict["train"], loss_fn, params)
            val_loss, val_acc = evaluate(model, loader_dict["val"], loss_fn, params)
            test_loss, test_acc = evaluate(model, loader_dict["test"], loss_fn, params)

            if model_scheduler is not None:
                model_scheduler.step(val_loss)
            logging(logger, tr_loss, tr_acc, val_loss, val_acc, test_loss, test_acc)

    for e in range(params["num_epochs"]):
        # print("Epoch", e)
        tr_loss, tr_acc = train(model, model_optimizer, loader_dict["train"], loss_fn, params, \
                                transformer, transformer_optimizer)
        val_loss, val_acc = evaluate(model, loader_dict["val"], loss_fn, params, transformer)
        test_loss, test_acc = evaluate(model, loader_dict["test"], loss_fn, params, transformer)
        
        if model_scheduler is not None:
            model_scheduler.step(val_loss)
        if transformer_schedular is not None:
            transformer_schedular.step(val_loss)

        logging(logger, tr_loss, tr_acc, val_loss, val_acc, test_loss, test_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

    result = {
        "train_acc": tr_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "train_loss": tr_loss,
        "val_loss": val_loss,
        "test_loss": test_loss,
        "best_val_acc": best_val_acc,
        "best_test_acc": best_test_acc
    }

    if transformer is not None:
        for name, param in transformer.state_dict().items():
            result[name] = param.data.cpu().numpy().tolist()

    for name, param in model.state_dict().items():
        result[name] = param.data.cpu().numpy().tolist()
    return result, logger