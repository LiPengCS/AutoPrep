import json
import logging
import os
import shutil
import torch
from matplotlib import pyplot as plt
import pandas as pd

def makedir(dir_list, file=None):
    save_dir = os.path.join(*dir_list)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if file is not None:
        save_dir = os.path.join(save_dir, file)
    return save_dir

def save_checkpoint(state, is_best, checkpoint):
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! " \
            "Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    torch.save(state, filepath)
    
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))

def load_checkpoint(checkpoint, model, optimizer=None):
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint, lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])
    return checkpoint

def plot_logger(logger, save_dir=None):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(logger["train_losses"], label="train_loss")
    plt.plot(logger["val_losses"], label="val_loss")
    plt.plot(logger["test_losses"], label="test_loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(logger["train_accs"], label="train_acc")
    plt.plot(logger["val_accs"], label="val_acc")
    plt.plot(logger["test_accs"], label="test_acc")
    plt.legend()

    if save_dir is None:
        plt.show()
    else:
        plt.savefig(os.path.join(save_dir, "history.png"))

def load_df(file_path, dataset_info):
    df = pd.read_csv(file_path)
    if 'categorical_variables' in dataset_info.keys():
        categories = dataset_info['categorical_variables']
        for cat in categories:
            if cat in df.columns:
                df[cat] = df[cat].astype(str).replace('nan', np.nan) 
    if "drop_variables" in dataset_info.keys():
        df = df.drop(columns=dataset_info["drop_variables"])
    return df

def load_info(info_dir):
    info_path = os.path.join(info_dir, "info.json")
    with open(info_path) as info_data:
        info = json.load(info_data)
    return info

def load_data(data_dir, dataset, keepna=True):
    # load info dict
    dataset_dir = os.path.join(data_dir, dataset)
    info = load_info(dataset_dir)

    file_path = os.path.join(dataset_dir, "data.csv")
    data = load_df(file_path, info)

    if not keepna:
        data = data.dropna().reset_index(drop=True)

    label_column = info["label"]
    feature_column = [c for c in data.columns if c != label_column]
    X = data[feature_column]
    y = data[[label_column]]
    return X, y

def save_result(result, logger, save_dir):
    with open(os.path.join(save_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=4)
    
    plot_logger(logger, save_dir)


