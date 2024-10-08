import tempfile

import pandas as pd
import mlflow
from mlflow import MlflowClient


def log_df_as_csv(df: pd.DataFrame, name: str) -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        filename = f"{tempdir}/{name}.csv"
        df.to_csv(filename, index=False)
        mlflow.log_artifact(filename, artifact_path="data")


def download_csv_as_df(run_id: str, name: str) -> pd.DataFrame:
    client = MlflowClient()
    with tempfile.TemporaryDirectory() as tempdir:
        filename = f"{tempdir}/data/{name}.csv"
        client.download_artifacts(
            run_id=run_id, path=f"data/{name}.csv", dst_path=tempdir
        )
        return pd.read_csv(filename)