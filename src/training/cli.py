import logging

import click

from src.utils.click import SpecialHelpOrder


log = logging.getLogger(__name__)


@click.group(cls=SpecialHelpOrder)
def cli():
    """Training pipeline tasks"""


@cli.command(
    help="A dummy task",
    help_priority=1,
)
@click.option(
    "--synthetic/--no-synthetic",
    "-s/-ns",
    "use_synthetic",
    is_flag=True,
    default=True,
    show_default=True,
    help="Whether to use " "synthetic data to " "run the step",
)
def run(use_synthetic):
    print(f"hello world: {use_synthetic}")
