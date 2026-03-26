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

See `config/mlflow_config_readme.txt` for a full description of all config JSON fields.

A complete working example lives in `src/tests/example/main.py`. You can run it with:

```bash
python -m src.tests.example.main
```

The example simulates 7 training epochs with fake accuracy and loss values, keeps only the top 2 models (`number_of_models_to_track=2`, objective `max`), and then loads the best model for an evaluation run. Below is a walkthrough of what it does.

### 1. Training run

```python
import torch.nn as nn
from src.config import MLFlowConfig
from src.tracker import MLFlowTracker, Model

# Simulated metrics for 7 epochs
accuracy = [0.1, 0.4, 0.35, 0.6, 0.8, 0.90, 0.85]
loss = [0.9, 0.8, 0.75, 0.6, 0.5, 0.4, 0.35]

config = MLFlowConfig.from_json('src/tests/example/mlflow_config.json')
client = config.apply(run_name="Training Run")

tracker = MLFlowTracker(client=client, config=config, min_or_max="max")

# Log global configs (Optuna search space) and dataset path before training
tracker.log_config(config_path="config/config_model_randlanet_0.json", save_config_for_model=False, save_as_parameters=True)
tracker.log_config(config_path="config/config_train_randlanet.json", save_config_for_model=False, save_as_parameters=False)
tracker.log_dataset(path="config/wynik 1.las")

# Training loop — only models that beat the current best accuracy are logged
for idx, acc in enumerate(accuracy):
    model = Model(
        model=nn.Sequential(nn.Linear(2, 1)),
        metrics={"accuracy": acc, "loss": loss[idx]},
        metrics_art={"cm": [[50, 2], [1, 47]]},
        configs=["config/config_model_randlanet_1.json", "config/config_train_single_randlanet.json"],
        best_val=acc
    )
    tracker.log_training(model=model, model_name="Resnet", number_of_models_to_track=2, step=idx)

config.end_run()
```

With `number_of_models_to_track=2` and `min_or_max="max"`, the tracker keeps only the 2 models with the highest accuracy. When a third model qualifies, the worst of the three is evicted (deleted from disk and MLflow).

### 2. Evaluation run

```python
config = MLFlowConfig.from_json('src/tests/example/mlflow_config.json')
client = config.apply(run_name="Evaluation Run")

eval_tracker = MLFlowTracker(client=client, config=config, min_or_max=None)
eval_tracker.log_dataset(path="config/wynik 1.las")

# Load a model saved during training by its full name
loaded_model, config_paths = eval_tracker.load_model(model_name="Resnet_2026-03-25_15-00-35_4")

# Log evaluation results — no duplicate model file is saved, only the model name is recorded
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
| `metrics_art` | `dict` | Non-scalar metrics logged as CSVs. Each value must be convertible to a 2D pandas DataFrame. Supported types: `list[list]` (e.g. a confusion matrix array from `sklearn`), `list[dict]` (records with named columns), `dict[str, list]` (column-oriented), or `np.ndarray`. Each key becomes a separate CSV artifact. |
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
