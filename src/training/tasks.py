import logging
from typing import Tuple, List, Dict

import mlflow
import numpy as np
import pandas as pd
from fairlearn.metrics import demographic_parity_difference
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_percentage_error,
    mean_absolute_error,
)
from sklearn.model_selection import train_test_split

from src import ROOT_DIR
from src.utils.io import read_yaml
from src.utils.mlflow import log_df_as_csv

logger = logging.getLogger(__name__)


def get_risky_features_list(
    feature_list: List[str], feature_catalogue: Dict
) -> List[str]:
    return [
        feature for feature in feature_list if feature_catalogue[feature]["is_risky"]
    ]


def get_demo_features_list(
    feature_list: List[str], feature_catalogue: Dict
) -> List[str]:
    return [
        feature for feature in feature_list if feature_catalogue[feature]["is_demo"]
    ]


def load_training_data() -> (
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict, Dict]
):
    """Train model on first"""
    logger.info("[+] Loading training data")
    X = pd.read_csv(ROOT_DIR / "data" / "feature_store.csv", parse_dates=["month"])
    X = X[X["month"] == pd.Timestamp("2024-01-01")].drop(columns=["month"])
    y = pd.read_csv(ROOT_DIR / "data" / "label_store.csv", parse_dates=["month"])
    y = y.loc[y["month"] == pd.Timestamp("2024-01-01"), "MEDV"]

    label_store_catalogue = read_yaml(ROOT_DIR / "data" / "label_store_catalogue.yaml")[
        "schema"
    ]
    label_store_catalogue = {y.name: label_store_catalogue[y.name]}
    feature_store_catalogue = read_yaml(
        ROOT_DIR / "data" / "feature_store_catalogue.yaml"
    )["schema"]
    feature_store_catalogue = {k: feature_store_catalogue[k] for k in X.columns}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.33
    )

    mlflow.log_param("Dataset size-Train", len(y_train))
    mlflow.log_param("Dataset size-Test", len(y_test))
    mlflow.log_param("Number of features", X.shape[1])
    mlflow.log_param(
        "Number of risky features",
        len(get_risky_features_list(X.columns, feature_store_catalogue)),
    )
    mlflow.log_param(
        "Number of demographic features",
        len(get_demo_features_list(X.columns, feature_store_catalogue)),
    )
    mlflow.log_dict(feature_store_catalogue, "input_data.yml")
    mlflow.log_dict(label_store_catalogue, "output_data.yml")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_store_catalogue,
        label_store_catalogue,
    )


def train_model(X, y):
    logger.info("[+] Training model")
    model = LinearRegression()
    model.fit(X, y)
    return model


def score_model(model, X_train, y_train, X_test, y_test, feature_catalogue):
    """Score model on validation set, report performance and fairness metrics"""
    logger.info("[+] Scoring trained model")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    input_example = X_train.head(5)

    # Performance -------------------------------------------------
    # Scores
    r2_test = r2_score(y_test, y_pred_test)
    err_test = mean_absolute_error(y_test, y_pred_test)
    err_perc_test = mean_absolute_percentage_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    err_train = mean_absolute_error(y_train, y_pred_train)
    err_perc_train = mean_absolute_percentage_error(y_train, y_pred_train)

    # Graphs
    df_to_plot = pd.DataFrame({"y_test": y_test, "y_pred_test": y_pred_test})
    df_to_plot["residual"] = np.abs(df_to_plot["y_test"] - df_to_plot["y_pred_test"])
    # residuals distribution
    f1, ax1 = plt.subplots()
    df_to_plot["residual"].hist(ax=ax1)
    # prediction scatter plot
    f2, ax2 = plt.subplots()
    df_to_plot.plot(kind="scatter", x="y_test", y="y_pred_test", ax=ax2)

    # Fairness & biases -------------------------------------------------
    # scores (binarize all continuous demo variables and run metrics)
    vars_to_check = get_demo_features_list(X_test.columns, feature_catalogue)
    vars_to_check_groups = (X_test[vars_to_check] > X_test[vars_to_check].median()) * 1
    y_pred_test_bin = (y_pred_test > y_test.median()) * 1
    y_test_bin = (y_test > y_test.median()) * 1
    for v in vars_to_check:
        dpd = demographic_parity_difference(
            y_test_bin, y_pred_test_bin, sensitive_features=vars_to_check_groups[v]
        )
        mlflow.log_metric(f"DPD-variable-{v}", dpd)

    # Saving all metrics to mlflow along with model ---------------------
    mlflow.sklearn.log_model(model, "model", input_example=input_example)
    mlflow.log_metric("R2-train", r2_train)
    mlflow.log_metric("R2-test", r2_test)
    mlflow.log_metric("MAE-train", err_train)
    mlflow.log_metric("MAE-test", err_test)
    mlflow.log_metric("MAPE-train", err_perc_train)
    mlflow.log_metric("MAPE-test", err_perc_test)

    # Saving all graphs to mlflow along with model ----------------------
    mlflow.log_figure(f1, "residuals_distribution.png")
    mlflow.log_figure(f2, "prediction_scatter.png")

    # Saving test data (later used to measure drift) --------------------
    train_dataset = mlflow.data.from_pandas(pd.concat([X_train, y_train], axis=1))
    test_dataset = mlflow.data.from_pandas(pd.concat([X_test, y_test], axis=1))
    mlflow.log_input(dataset=train_dataset, context="training")
    mlflow.log_input(dataset=test_dataset, context="testing")
    log_df_as_csv(X_train, name="X_train")
    log_df_as_csv(y_train, name="y_train")
