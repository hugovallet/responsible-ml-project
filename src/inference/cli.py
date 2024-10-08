import logging

import click

from src.inference.tasks import load_inference_data, load_model, run_inference, score_inference
from src.utils.click import SpecialHelpOrder


log = logging.getLogger(__name__)


@click.group(cls=SpecialHelpOrder)
def cli():
    """Inference pipeline tasks"""


@cli.command(
    help="Run the inference pipeline",
    help_priority=1,
)
def run():
    X = load_inference_data()
    model = load_model()
    y_pred = run_inference(model, X)
    score_inference(y_pred)