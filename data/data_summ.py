import pandas as pd
import os
import json
import numpy as np
from shutil import copy

data_dir = "zhiyi_mv"
datasets = [d for d in os.listdir(data_dir) if d[0] != "."]

summ = []
for d in datasets:
    data = pd.read_csv(os.path.join(data_dir, d, "{}.csv".format(d)), dtype=str)
    data.replace({'?': np.nan}, regex=False,inplace=True)
    data.to_csv(os.path.join(data_dir, d, "data.csv"), index=False)


#     N = len(data)
#     m = data.shape[1] - 1


#     percent_mv = data.isnull().values.mean()

#     n_cat_mv = data.select_dtypes(exclude="number").isnull().values.any(axis=0).sum()
#     n_num_mv = data.select_dtypes(include="number").isnull().values.any(axis=0).sum()

#     summ.append([d, N, m, percent_mv, n_num_mv, n_cat_mv])
# pd.DataFrame(summ, columns=["dataset", "# rows", "# columns", "percent_mv", "n_num_mv", "n_cat_mv"]).to_csv("zhiyi_mv_dataset.csv", index=False)