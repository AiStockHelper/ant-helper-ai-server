import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def build_dataset(df: pd.DataFrame, window=252):
    features = df.values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X, y = [], []

    for i in range(len(features_scaled) - window - 1):
        X.append(features_scaled[i:i + window])
        y.append(features_scaled[i + window][df.columns.get_loc("Close")])

    return np.array(X), np.array(y), scaler