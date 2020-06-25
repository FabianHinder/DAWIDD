# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def read_data_electricity_market(foldername="data/"):
    df = pd.read_csv(foldername + "elecNormNew.csv")
    data = df.values
    X, y = data[:, 1:-1], data[:, -1]

    # Set x,y as numeric
    X = X.astype(float)
    label = ["UP", "DOWN"]
    le = LabelEncoder()
    le.fit(label)
    y = le.transform(y)

    return X, y

def read_data_weather(foldername="data/weather/"):
    df_labels = pd.read_csv(foldername + "NEweather_class.csv")
    y = df_labels.values.flatten()

    df_data = pd.read_csv(foldername + "NEweather_data.csv")
    X = df_data.values

    return X, y 


def read_data_forest_cover_type(foldername="data/"):
    df = pd.read_csv(foldername + "forestCoverType.csv")
    data = df.values
    X, y = data[:, 1:-1], data[:, -1]

    return X, y
