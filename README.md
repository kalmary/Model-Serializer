# Model-Serializer

A PyTorch model tracking library built on MLflow. Keeps the top-N best-performing models during training and automatically evicts the worst when the limit is exceeded.

## Features

- Tracks the top-N models by a configurable objective (min or max)
- Automatically evicts the worst model (filesystem + MLflow artifacts) when the limit is exceeded
- Logs models (`.pt`), scalar metrics, CSV artifacts, and config files per model
- Config loaded from a simple JSON file

## Requirements

- Python 3.13+
- `uv` (package manager)
- Dependencies: `mlflow`, `torch`, `pandas`, `numpy`

## Installation

```bash
uv sync
```

## Quick Start

**1. Create a config JSON:**

```json
{
  "tracking_uri": "sqlite:///mlflow.db",
  "artifact_dir": "/path/to/artifacts",
  "experiment_name": "my-experiment",
  "run_name": "run-1",
  "models_dir": "models"
}
```

**2. Training loop:**

```python
import torch.nn as nn
from src.config import MLFlowConfig
from src.tracker import MLFlowTracker, Model

config = MLFlowConfig.from_json("mlflow_config.json")
client = config.apply()

tracker = MLFlowTracker(
    client=client,
    config=config,
    number_of_models_to_track=3,
    min_or_max="max"   # or "min" for loss-based objectives
)

for epoch, (acc, loss) in enumerate(training_results):
    model = Model(
        model=my_nn,
        metrics={"accuracy": acc, "loss": loss},
        metrics_art={"confusion_matrix": [[50, 2], [1, 47]]},
        config="config/model_config.json",
        best_val=acc
    )
    tracker.log_training(model=model, model_name="MyModel", step=epoch)
```

## API Reference

### `MLFlowConfig`

| Method | Description |
|--------|-------------|
| `MLFlowConfig.from_json(path)` | Load config from a JSON file |
| `config.apply(run_name=None)` | Set up MLflow experiment, start a run, return `MlflowClient` |

### `MLFlowTracker(client, config, number_of_models_to_track, min_or_max)`

| Method | Description |
|--------|-------------|
| `log_training(model, model_name, step)` | Main entry point — logs model if it beats the current best |
| `log_config(config_path, ...)` | Log a JSON config as artifact and/or MLflow parameters |
| `log_dataset(path)` | Log dataset path as an MLflow parameter |
| `log_metrics(metrics, step, ...)` | Log scalar metrics dict |
| `log_metrics_artifact(metrics, ...)` | Log structured metrics as CSV artifacts |

### `Model` dataclass

| Field | Type | Description |
|-------|------|-------------|
| `model` | `nn.Module` | PyTorch model |
| `metrics` | `dict` | Scalar metrics, e.g. `{"accuracy": 0.9}` |
| `metrics_art` | `dict` | Non-scalar metrics logged as CSVs, e.g. `{"cm": [[50, 2], [1, 47]]}` |
| `config` | `str \| Path` | Path to the model config file |
| `best_val` | `float` | Value used to compare models (mapped to `min_or_max` objective) |

### Config JSON fields

| Field | Required | Description |
|-------|----------|-------------|
| `tracking_uri` | Yes | MLflow tracking URI (SQLite or server) |
| `artifact_dir` | Yes | Root directory for storing artifacts |
| `experiment_name` | Yes | MLflow experiment name |
| `run_name` | Yes | MLflow run name |
| `models_dir` | No | Subdirectory for model files (default: `"models"`) |
| `run_tags` | No | Dict of tags attached to the run |
| `experiment_tags` | No | Dict of tags attached to the experiment |