import logging
import os
from time import sleep
from typing import Tuple, List

import mlflow
import pandas as pd
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.renderers.html_widgets import WidgetSize
from evidently.report import Report
from evidently.ui.base import Project
from evidently.ui.dashboards import (
    DashboardPanelPlot,
    ReportFilter,
    PanelValue,
    PlotType,
)
from evidently.ui.errors import ProjectNotFound
from evidently.ui.workspace import Workspace

from src import ROOT_DIR
from src.data.constants import FIRST_DATE
from src.data.tasks import load_feature_df
from src.inference.constants import EVIDENTLY_WS
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


def _get_evidently_tracker(model_name) -> Project:
    """Makes sure there's an Evidently dashboard associated with model in workspace"""
    ws = Workspace(EVIDENTLY_WS)
    all_projects_in_ws = {p.name: p for p in ws.list_projects()}
    if model_name not in all_projects_in_ws:
        # if this is a new model let's auto create an associated Evidently tracker
        ml_flow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        model_full_uri = f"{ml_flow_tracking_uri}#/models/{model_name}"
        project = ws.create_project(
            name=model_name,
            description=f"Dashboard for project associated with model '{model_name}' \n"
                        f"See: {model_full_uri}",
        )
        project.dashboard.add_panel(
            DashboardPanelPlot(
                title="Share of drifting features",
                filter=ReportFilter(metadata_values={}, tag_values=[]),
                values=[
                    PanelValue(
                        metric_id="DatasetDriftMetric",
                        field_path="share_of_drifted_columns",
                        legend="share",
                    ),
                ],
                plot_type=PlotType.LINE,
                size=WidgetSize.FULL,
            ),
            tab="Summary",
        )
        project.save()
        return project
    else:
        return all_projects_in_ws.get(model_name)


def load_inference_data(cols: List[str] = None) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """
    Load the right data for inference:
    1. select the latest pivot in feature store
    2. keep only the columns required by the model we will run
    """
    inference_month = _infer_prediction_month()
    logger.info(f"[+] Loading inference data for month {inference_month}")
    X = load_feature_df(pivot=inference_month)
    if cols is not None:
        X = X[cols]

    return X, inference_month


def load_model(model_name: str):
    logger.info(f"[+] Loading latest production model '{model_name}' from MLFlow")
    model_uri = f"models:/{model_name}/latest"
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


def score_inference(X, y_pred, model_name, model_infos, ref_month):
    logger.info(f"[+] Reporting drift for inference month {ref_month}")
    model_uri = f"models:/{model_name}/latest"
    X_test = download_csv_as_df(run_id=model_infos.run_id, name="X_test")
    y_pred_test = download_csv_as_df(run_id=model_infos.run_id, name="y_pred_test")
    reference_data = pd.concat([X_test, y_pred_test], axis=1)
    current_data = pd.concat([X, y_pred], axis=1)

    ws = Workspace(EVIDENTLY_WS)
    project = _get_evidently_tracker(model_name)
    drift_report = Report(
        metrics=[DataDriftPreset(), TargetDriftPreset()],
        model_id=model_uri,
        timestamp=ref_month,
    )
    drift_report.run(reference_data=reference_data, current_data=current_data)
    try:
        ws.add_report(project.id, drift_report)
    except ProjectNotFound:
        # catch weird Evidently error when dashboard to be created
        pass
    except Exception as e:
        raise e
