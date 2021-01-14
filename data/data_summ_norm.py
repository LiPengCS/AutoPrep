import pandas as pd
import os
import json
import numpy as np
from shutil import copy
from collections import Counter

data_dir = "normalization_datasets"
datasets = [d for d in os.listdir(data_dir) if d[0] != "."]

summ = []
for d in datasets:
    data = pd.read_csv(os.path.join(data_dir, d, "data.csv"))
    N = len(data)
    m = data.shape[1] - 1
    with open(os.path.join(data_dir, d, "info.json")) as f:
        info = json.load(f)
    y = data[info["label"]]
    count = Counter(y)
    n_major = count.most_common(1)[0][1]
    percent_major = n_major / len(y)
    summ.append([d, N, m, percent_major])
pd.DataFrame(summ, columns=["dataset", "# rows", "# columns", "percent_major"]).to_csv("norm_dataset.csv", index=False)