import logging
from typing import Tuple

import mlflow
import pandas as pd
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.report import Report
from evidently.ui.workspace import Workspace

from src import ROOT_DIR
from src.data.constants import FIRST_DATE
from src.data.tasks import load_feature_df
from src.inference.constants import EVIDENTLY_WS, EVIDENTLY_PROJECT_ID
from src.utils.mlflow import download_csv_as_df

logger = logging.getLogger(__name__)


def _infer_prediction_month():
    """Automatically increments the prediction month depending on previous runs. Allows us to simulate a realistic
    production environment where inference has to run monthly."""
    predictions = pd.read_csv(
        ROOT_DIR / "data" / "prediction_store.csv", parse_dates=["month"]
    )
    if predictions.empty:
        return FIRST_DATE + pd.DateOffset(months=1)
    else:
        return max(predictions["month"]) + pd.DateOffset(months=1)


def load_inference_data() -> Tuple[pd.DataFrame, pd.Timestamp]:
    inference_month = _infer_prediction_month()
    logger.info(f"[+] Loading inference data for month {inference_month}")
    X = load_feature_df(pivot=inference_month)

    return X, inference_month


def load_model():
    logger.info("[+] Loading latest production model from MLFlow")
    model_name = "linear-model-on-boston-prod"
    model_version = "latest"
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    model_infos = mlflow.models.get_model_info(model_uri)

    return model, model_infos


def run_inference(model, model_infos, X, ref_month) -> pd.DataFrame:
    logger.info(f"[+] running inference using model from run {model_infos.run_id}")
    y_pred = model.predict(X)
    y_pred = pd.DataFrame(y_pred, columns=["target"])
    prediction_df = pd.DataFrame(
        {"month": ref_month, "model_id": model_infos.run_id, "target": y_pred["target"]}
    )
    prediction_df.to_csv(
        ROOT_DIR / "data" / "prediction_store.csv", index=False, mode="a", header=False
    )

    return y_pred


def score_inference(X, y_pred, model_infos, ref_month):
    logger.info(f"[+] Reporting drift for inference month {ref_month}")
    model_name = "linear-model-on-boston-prod"
    model_version = "latest"
    model_uri = f"models:/{model_name}/{model_version}"
    X_test = download_csv_as_df(run_id=model_infos.run_id, name="X_test")
    y_pred_test = download_csv_as_df(run_id=model_infos.run_id, name="y_pred_test")
    reference_data = pd.concat([X_test, y_pred_test], axis=1)
    current_data = pd.concat([X, y_pred], axis=1)

    ws = Workspace(EVIDENTLY_WS)
    project = ws.get_project(EVIDENTLY_PROJECT_ID)
    drift_report = Report(
        metrics=[DataDriftPreset(), TargetDriftPreset()], model_id=model_uri
    )
    drift_report.run(reference_data=reference_data, current_data=current_data)
    ws.add_report(project.id, drift_report)
