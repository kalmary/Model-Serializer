#!/bin/bash
CONFIG_FILE="$(dirname "$0")/src/mlflow_config.json"
export MLFLOW_TRACKING_URI=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['tracking_uri'])")
exec uv run --with "mlflow[mcp]>=3.5.1" mlflow mcp run
