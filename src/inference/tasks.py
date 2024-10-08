import logging
from typing import Tuple

import mlflow
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from evidently.ui.workspace import Workspace

from src import ROOT_DIR
from src.utils.mlflow import download_csv_as_df

logger = logging.getLogger(__name__)


def _infer_prediction_month():
    """Automatically increments the prediction month depending on previous runs. Allows us to simulate a realistic
    production environment where inference has to run monthly."""
    predictions = pd.read_csv(ROOT_DIR / "data" / "prediction_store.csv", parse_dates=["month"])
    if predictions.empty:
        return pd.Timestamp("2024-02-01")
    else:
        return max(predictions["month"]) + pd.Timedelta(months=1)


def load_inference_data() -> Tuple[pd.DataFrame, pd.Timestamp]:
    inference_month = _infer_prediction_month()
    logger.info(f"[+] Loading inference data for month {inference_month}")
    X = pd.read_csv(ROOT_DIR / "data" / "feature_store.csv", parse_dates=["month"])
    X = X[X["month"] == inference_month]

    return X, inference_month


def load_model():
    model_name = "linear-model-on-boston-prod"
    model_version = "latest"
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    infos = mlflow.models.get_model_info(model_uri)
    X_train = download_csv_as_df(run_id=infos.run_id, name="X_train")
    y_train = download_csv_as_df(run_id=infos.run_id, name="y_train")

    return model, X_train, y_train


def run_inference(model, X):
    y_pred = model.predict(X)
    return y_pred


def score_inference(X, y_pred, X_train, y_train, ref_month):
    model_name = "linear-model-on-boston-prod"
    model_version = "latest"
    model_uri = f"models:/{model_name}/{model_version}"
    ws = Workspace("evidently/")
    project = ws.get_project("01926d6a-e2c9-7941-a7e1-2a6243f9a0b3")
    drift_report = Report(metrics=[DataDriftPreset()], model_id=model_uri)
    drift_report.run(reference_data=X_train, current_data=X)
    ws.add_report(project.id, drift_report)
