import logging

import click
from evidently import metrics
from evidently.renderers.html_widgets import WidgetSize
from evidently.ui.dashboards import (
    DashboardPanelPlot,
    ReportFilter,
    PanelValue,
    PlotType,
)
from evidently.ui.workspace import Workspace

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
def run():
    X, ref_month = load_inference_data()
    model, X_train, y_train = load_model()
    y_pred = run_inference(model, X)
    score_inference(X, y_pred, X_train, y_train, ref_month)


@cli.command(
    help="Start dashboard",
    help_priority=2,
)
def init_evidently_dashboard():
    ws = Workspace("evidently/")
    project = ws.get_project("01926d6a-e2c9-7941-a7e1-2a6243f9a0b3")
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Monthly inference Count",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            values=[
                PanelValue(
                    metric_id="DatasetSummaryMetric",
                    field_path=metrics.DatasetSummaryMetric.fields.current.number_of_rows,
                    legend="count",
                ),
            ],
            plot_type=PlotType.LINE,
            size=WidgetSize.FULL,
        ),
        tab="Summary",
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
