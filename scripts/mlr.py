#!/usr/bin/env python3

import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

TRAIN_DATA_PATH = Path("../data/train_data_csv")

bmc_engines = ["bmc2", "bmc3", "bmc3g", "bmc3r", "bmc3u", "bmc3s", "bmc3j"]
feature_cols = ["Var", "Cla", "Conf", "Learn"]
target_col = ["Time"]

X_dict: dict[str, pd.DataFrame] = {}
y_dict: dict[str, pd.DataFrame] = {}
model_dict: dict[str, LinearRegression] = {}
for bmc_engine in bmc_engines:
    X = pd.DataFrame({col: [] for col in feature_cols})
    y = pd.DataFrame({col: [] for col in target_col})

    ENGINE_TRAIN_DATA_PATH = TRAIN_DATA_PATH / bmc_engine
    csv_files = list(ENGINE_TRAIN_DATA_PATH.glob("*.csv"))
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        print(df.dtypes)

        for feature_col in feature_cols:
            df[feature_col] = pd.to_numeric(df[feature_col])
        df["Time"] = pd.to_numeric(df["Time"])
        df["del_T"] = df["Time"].diff()

        df = df.dropna(how="any")

        X = pd.DataFrame(pd.concat([X, df[feature_cols]], axis=0))
        y = pd.DataFrame(pd.concat([y, df[target_col]], axis=0))

    X_dict[bmc_engine] = X
    y_dict[bmc_engine] = y

    model_dict[bmc_engine] = LinearRegression()
    model_dict[bmc_engine].fit(X, y)

with open("../data/model.pkl", "wb") as pkl_file:
    pickle.dump(model_dict, pkl_file)
