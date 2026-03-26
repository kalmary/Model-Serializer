# Model-Serializer

A PyTorch model tracking library built on MLflow. Keeps the top-N best-performing models during training and automatically evicts the worst when the limit is exceeded.

## Features

- Tracks the top-N models by a configurable objective (min or max)
- Automatically evicts the worst model (filesystem + MLflow artifacts) when the limit is exceeded
- Logs models (`.pt`), scalar metrics, CSV artifacts, and config files per model
- Evaluation logging records which model was used without saving a duplicate file
- Load previously saved models by name for evaluation or inference
- Config loaded from a simple JSON file

## Requirements

- Python 3.13+
- `uv` (package manager)

## Installation

```bash
uv sync
```

Or with pip:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Create a config JSON

```json
{
  "tracking_uri": "sqlite:///mlflow.db",
  "artifact_dir": "/path/to/artifacts",
  "experiment_name": "my-experiment",
  "run_name": "run-1",
  "models_dir": "models"
}
```

See `config/mlflow_config_readme.txt` for a full description of all fields.

### 2. Training loop

```python
import torch.nn as nn
from src.config import MLFlowConfig
from src.tracker import MLFlowTracker, Model

config = MLFlowConfig.from_json("mlflow_config.json")
client = config.apply("Training Run")

tracker = MLFlowTracker(client=client, config=config, min_or_max="max")

# Log initial configs and dataset
tracker.log_config(config_path="config/model_config.json", save_config_for_model=False, save_as_parameters=True)
tracker.log_dataset(path="data/dataset.las")

for epoch in range(num_epochs):
    # ... train your model ...
    model = Model(
        model=my_nn,
        metrics={"accuracy": acc, "loss": loss},
        metrics_art={"cm": [[50, 2], [1, 47]]},
        configs=["config/model_config.json", "config/train_config.json"],
        best_val=acc
    )
    tracker.log_training(model=model, model_name="Resnet", number_of_models_to_track=2, step=epoch)

config.end_run()
```

### 3. Evaluation

```python
config = MLFlowConfig.from_json("mlflow_config.json")
client = config.apply("Evaluation Run")

eval_tracker = MLFlowTracker(client=client, config=config, min_or_max=None)
eval_tracker.log_dataset(path="data/dataset.las")

# Load a model saved during training by its full name
loaded_model, config_paths = eval_tracker.load_model(model_name="Resnet_2026-03-25_15-00-35_4")

# Run evaluation and log results (no duplicate model file is saved)
test_metrics = {"accuracy": 0.88, "loss": 0.38}
model = Model(
    model=loaded_model,
    metrics=test_metrics,
    metrics_art={"cm": [[50, 2], [1, 47]]},
    configs=config_paths,
    best_val=test_metrics["accuracy"]
)
eval_tracker.log_evaluation(model=model, model_name="Resnet_2026-03-25_15-00-35_4")
config.end_run()
```

## Running the Example

```bash
python -m src.tests.michal_fake_env.main
```

This creates 7 fake models with varying accuracy and tests the tracker with `number_of_models_to_track=2` using max objective, then runs an evaluation on the best model.

## API Reference

### `MLFlowConfig`

| Method | Description |
|--------|-------------|
| `MLFlowConfig.from_json(path)` | Load config from a JSON file |
| `config.apply(run_name=None)` | Set up MLflow experiment, start a run, return `MlflowClient` |
| `config.end_run()` | End the current MLflow run |

### `MLFlowTracker(client, config, min_or_max)`

| Method | Description |
|--------|-------------|
| `log_training(model, model_name, number_of_models_to_track, step)` | Log model if it beats the current best, evict worst if limit exceeded |
| `log_evaluation(model, model_name)` | Log evaluation metrics and configs (no model file saved, name recorded as parameter) |
| `load_model(model_name)` | Load a previously saved model by name, returns `(nn.Module, list[Path])` |
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
| `configs` | `list[str \| Path]` | Paths to config files associated with this model |
| `best_val` | `float` | Value used to compare models (mapped to `min_or_max` objective) |

### Config JSON fields

| Field | Required | Description |
|-------|----------|-------------|
| `tracking_uri` | Yes | MLflow tracking URI (SQLite or server) |
| `artifact_dir` | Yes | Root directory for storing artifacts |
| `experiment_name` | Yes | MLflow experiment name |
| `run_name` | Yes | Default MLflow run name (can be overridden in `apply()`) |
| `models_dir` | No | Subdirectory for model files (default: `"models"`) |
| `run_tags` | No | Dict of tags attached to the run |
| `experiment_tags` | No | Dict of tags attached to the experiment |
