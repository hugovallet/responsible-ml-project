import logging

import click
from mlflow.types import Schema

from src.inference.tasks import (
    load_inference_data,
    load_model,
    run_inference,
    score_inference,
)
from src.utils.click import SpecialHelpOrder


log = logging.getLogger(__name__)


@click.group(cls=SpecialHelpOrder)
def cli():
    """Inference pipeline tasks"""


@cli.command(
    help="Run the inference pipeline",
    help_priority=1,
)
@click.option("-m", "--model_name", "model_name")
def run(model_name):
    # retrieve latest model version from MLFlow
    model, model_infos = load_model(model_name)
    # load inference data
    X, ref_month = load_inference_data(
        cols=Schema.input_names(model.metadata.signature.inputs)
    )
    # run model on data
    y_pred = run_inference(
        model=model, model_infos=model_infos, X=X, ref_month=ref_month
    )
    # check and log drift
    score_inference(
        X=X,
        y_pred=y_pred,
        model_name=model_name,
        model_infos=model_infos,
        ref_month=ref_month,
    )
