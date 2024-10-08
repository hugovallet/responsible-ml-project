import logging

import click
import mlflow

from src.training.tasks import load_training_data, train_model, score_model
from src.utils.click import SpecialHelpOrder


log = logging.getLogger(__name__)


@click.group(cls=SpecialHelpOrder)
def cli():
    """Training pipeline tasks"""


@cli.command(
    help="Runs the training pipeline",
    help_priority=1,
)
def run():
    mlflow.end_run()
    with mlflow.start_run():
        (
            X_train,
            X_test,
            y_train,
            y_test,
            feature_catalogue,
            label_catalogue,
        ) = load_training_data()
        model = train_model(X_train, y_train)
        score_model(model, X_train, y_train, X_test, y_test, feature_catalogue)
