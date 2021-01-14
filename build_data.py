import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

def split(X, y, val_ratio=0.2, test_ratio=0.2, random_state=1):
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_ratio, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_ratio, random_state=random_state)
    return X_train, y_train, X_val, y_val, X_test, y_test 

def build_data(X, y, normalize="standard", random_state=1):
    num_features = X.select_dtypes(include='number').columns
    cat_features = X.select_dtypes(exclude='number').columns

    label_enc = LabelEncoder()
    onehot_enc = OneHotEncoder(sparse=False, handle_unknown='ignore')

    y_enc = label_enc.fit_transform(y.values.ravel())

    if len(cat_features) > 0:
        X_num = X[num_features].values

        X_cat_enc = pd.get_dummies(X[cat_features], dummy_na=True, dtype=float)

        for cat_c in cat_features:
            is_nan = X_cat_enc[cat_c + "_nan"].values == 1
            enc_features = [c for c in X_cat_enc.columns if c.startswith("{}_".format(cat_c)) and c!= cat_c + "_nan"]

            for enc_c in enc_features:
                X_cat_c = X_cat_enc[enc_c].values
                X_cat_c[is_nan] = np.nan
                X_cat_enc[enc_c] = X_cat_c

        X_cat_enc = X_cat_enc.drop(columns=[c + "_nan" for c in cat_features])

        X_enc = np.hstack([X_num, X_cat_enc.values])
    else:
        X_enc = X.values

    X_train, y_train, X_val, y_val, X_test, y_test = split(X_enc, y_enc, random_state=random_state)

    if normalize == None:
        return X_train, y_train, X_val, y_val, X_test, y_test

    if normalize == "standard":
        normalizer = StandardScaler()
    else:
        normalizer = MinMaxScaler()
    normalizer.fit(X_train)
    X_train = normalizer.transform(X_train)
    X_val = normalizer.transform(X_val)
    X_test = normalizer.transform(X_test)
    return X_train, y_train, X_val, y_val, X_test, y_test