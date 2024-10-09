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

from src.inference.constants import EVIDENTLY_PROJECT_ID, EVIDENTLY_WS
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
    model, model_infos = load_model()
    y_pred = run_inference(
        model=model, model_infos=model_infos, X=X, ref_month=ref_month
    )
    score_inference(X=X, y_pred=y_pred, model_infos=model_infos, ref_month=ref_month)


@cli.command(
    help="Start dashboard",
    help_priority=2,
)
def init_dashboard():
    ws = Workspace(EVIDENTLY_WS)
    project = ws.get_project(EVIDENTLY_PROJECT_ID)
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
