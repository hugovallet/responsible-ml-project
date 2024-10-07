#!/usr/bin/env bash
set -eou pipefail

evidently ui \
    --workspace ./workspace \
    --host 0.0.0.0 \
    --port $EVIDENTLY_PORT
