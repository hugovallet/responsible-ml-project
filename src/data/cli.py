import logging

import click

import src.data.tasks as tasks
from src.utils.click import SpecialHelpOrder


log = logging.getLogger(__name__)


@click.group(cls=SpecialHelpOrder)
def cli():
    """Data prep tasks tasks"""


@cli.command(
    help="Build label and feature stores",
    help_priority=1,
)
def build_label_feature_stores():
    tasks.build_label_feature_stores()


@cli.command(
    help="Empty the prediction store",
    help_priority=2,
)
def reset_prediction_store():
    tasks.reset_prediction_store()
