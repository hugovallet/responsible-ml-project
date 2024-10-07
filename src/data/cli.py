import logging

import click

from src.data.tasks import build_stores
from src.utils.click import SpecialHelpOrder


log = logging.getLogger(__name__)


@click.group(cls=SpecialHelpOrder)
def cli():
    """Data prep tasks tasks"""


@cli.command(
    help="Build label and feature stores",
    help_priority=1,
)
def build_label_feature_store():
    build_stores()
