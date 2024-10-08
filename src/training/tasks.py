import logging
from typing import Tuple

import mlflow
import pandas as pd
from fairlearn.metrics import demographic_parity_difference
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from src import ROOT_DIR

logger = logging.getLogger(__name__)


def load_training_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train model on first """
    logger.info("[+] Loading training data")
    X = pd.read_csv(ROOT_DIR / "data" / "feature_store.csv", parse_dates=["month"])
    X = X[X["month"] == pd.Timestamp("2024-01-01")].drop(columns=["month"])
    y = pd.read_csv(ROOT_DIR / "data" / "label_store.csv", parse_dates=["month"])
    y = y.loc[y["month"] == pd.Timestamp("2024-01-01"), "MEDV"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.33)

    mlflow.log_param("Dataset size-Train", len(y_train))
    mlflow.log_param("Dataset size-Test", len(y_test))
    mlflow.log_dict({"A": {"type": float, "description": "lorem ipsum"}}, "input_data.yml")
    mlflow.log_dict({"CPMV_pred": {"type": float, "description": "lorem ipsum"}}, "output_data.yml")

    return X_train, X_test, y_train, y_test


def train_model(X, y):
    logger.info("[+] Training model")
    model = LinearRegression()
    model.fit(X, y)
    return model


def score_model(model, X_train, y_train, X_test, y_test):
    """Score model on validation set, report performance and fairness metrics"""
    logger.info("[+] Scoring trained model")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    # performance scores
    r2_test = r2_score(y_test, y_pred_test)
    err_test = mean_absolute_error(y_test, y_pred_test)
    err_perc_test = mean_absolute_percentage_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    err_train = mean_absolute_error(y_train, y_pred_train)
    err_perc_train = mean_absolute_percentage_error(y_train, y_pred_train)
    # fairness and bias scores

    # Saving all metrics to mlflow along with model
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_metric("R2-train", r2_train)
    mlflow.log_metric("R2-test", r2_test)
    mlflow.log_metric("MAE-train", err_train)
    mlflow.log_metric("MAE-test", err_test)
    mlflow.log_metric("MAPE-train", err_perc_train)
    mlflow.log_metric("MAPE-test", err_perc_test)