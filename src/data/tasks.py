import numpy as np
import pandas as pd
from fairlearn.datasets import fetch_boston

from src import ROOT_DIR
from src.data.constants import FIRST_DATE, DRIFT_SIZE, N_DRIFTING_FEATURES


def __add_gaussian_noise(data, percent: float):
    std = data.std(axis=0)
    min_ = data.min(axis=0)
    noise = (
        pd.DataFrame(np.random.normal(0, std, size=data.shape), columns=data.columns)
        * percent
    )

    return np.clip(data + noise, a_min=min_, a_max=None, axis=1)


def build_label_feature_stores():
    """
    For this Proof-of-concept, we use the boston a data source notoriously famous for containing some serious ethical
    issues. We modify that dataset to also include a temporal drift in some variables, imagining that the same dataset
    was collected at multiple points in time. This allows us to test drift detection algorithms.
    """
    # initialise first month
    features, label = fetch_boston(return_X_y=True)
    features = features.drop(columns=["CHAS", "RAD"])
    label = label.to_frame()

    label_store = label.copy()
    feature_store = features.copy()
    feature_store["month"] = FIRST_DATE
    label_store["month"] = FIRST_DATE
    drifting_features = np.random.choice(
        features.columns, N_DRIFTING_FEATURES, replace=False
    )

    for month in range(1, 8):
        # for the first 3 months, data stays roughly same (no drift at all)
        month_features = features.copy()
        month_label = label.copy()
        month_features["month"] = FIRST_DATE + pd.DateOffset(months=month)
        month_label["month"] = FIRST_DATE + pd.DateOffset(months=month)
        # for the next 3 months, the drifting features see their mean increase by 2 times std, huge drift! (while label
        # stays the same)
        if month >= 4:
            for f in drifting_features:
                month_features[f] += DRIFT_SIZE * month_features[f].std(axis=0)

        feature_store = pd.concat([feature_store, month_features])
        label_store = pd.concat([label_store, month_label])

    feature_store.to_csv(ROOT_DIR / "data" / "feature_store.csv", index=False)
    label_store.to_csv(ROOT_DIR / "data" / "label_store.csv", index=False)


def reset_prediction_store():
    """In this proof pf concept, as we are not using real data, when running multiple times the inference pipeline
    you will run out of data. This command allows to restart the process from first month as if you never made any
    inferences."""
    empty_store_df = pd.DataFrame(columns=["month", "model_id", "target"])
    empty_store_df.to_csv(ROOT_DIR / "data" / "prediction_store.csv", index=False)


def load_feature_df(pivot: pd.Timestamp) -> pd.DataFrame:
    """Load the slice of store corresponding to selected pivot month"""
    X = pd.read_csv(ROOT_DIR / "data" / "feature_store.csv", parse_dates=["month"])
    if pivot not in X["month"].unique():
        raise ValueError(f"Pivot month {pivot} not available in dataset")
    X = X[X["month"] == pd.Timestamp(pivot)].drop(columns=["month"])

    return X


def load_label_df(pivot: pd.Timestamp) -> pd.DataFrame:
    """Load the slice of store corresponding to selected pivot month"""
    y = pd.read_csv(ROOT_DIR / "data" / "label_store.csv", parse_dates=["month"])
    if pivot not in y["month"].unique():
        raise ValueError(f"Pivot month {pivot} not available in dataset")
    y = y.loc[y["month"] == pd.Timestamp(pivot), "MEDV"]

    return y
