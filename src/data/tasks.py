import numpy as np
import pandas as pd
from fairlearn.datasets import fetch_boston

from src import ROOT_DIR


def __add_gaussian_noise(data, percent: float):
    std = data.std(axis=0)
    min_ = data.min(axis=0)
    noise = (
        pd.DataFrame(np.random.normal(0, std, size=data.shape), columns=data.columns)
        * percent
    )
    noise = np.clip(noise, a_min=min_, a_max=None, axis=1)
    return data + noise


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
    first_date = pd.Timestamp("2024-01-01")

    label_store = label.copy()
    feature_store = features.copy()
    feature_store["month"] = first_date
    label_store["month"] = first_date
    drifting_features = np.random.choice(features.columns, 3, replace=False)

    for month in range(1, 8):
        # for the first 3 months, data stays roughly the same (first month with little Gaussian noise)
        month_features = __add_gaussian_noise(features.copy(), 0.0)
        month_label = __add_gaussian_noise(label.copy(), 0.0)
        month_features["month"] = first_date + pd.DateOffset(months=month)
        month_label["month"] = first_date + pd.DateOffset(months=month)
        # for the next 3 months, the drifting features see their mean increase by 3 times std, huge drift! (while label
        # stays the same)
        if month >= 4:
            for f in drifting_features:
                month_features[f] += 2 * month_features[f].std(axis=0)

        feature_store = pd.concat([feature_store, month_features])
        label_store = pd.concat([label_store, month_label])

    feature_store.to_csv(ROOT_DIR / "data" / "feature_store.csv", index=False)
    label_store.to_csv(ROOT_DIR / "data" / "label_store.csv", index=False)


def reset_prediction_store():
    """In this proof pf concept, as we are not using real data, when running multiple times the inference pipeline
    you will run out of data. This command allows to restart the process from first month as if you never made any
    inferences."""
    empty_store_df = pd.DataFrame(columns=["month", "model_id", "prediction"])
    empty_store_df.to_csv(ROOT_DIR / "data" / "prediction_store.csv", index=False)
