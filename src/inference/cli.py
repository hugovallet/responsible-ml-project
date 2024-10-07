import logging

import click

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
    pass
