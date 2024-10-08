#!/usr/bin/env bash
set -eou pipefail

mlflow server \
    --app-name basic-auth \
    --port $MLFLOW_PORT \
    --host 0.0.0.0 \
    --backend-store-uri $MLFLOW_TRACKING_URI \
    --default-artifact-root $MLFLOW_ARTIFACT_ROOT
