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
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

See `config/mlflow_config_readme.txt` for a full description of all config JSON fields.

First configure the mlflow_config.json file inside the example folder.
 
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

## Viewing the MLflow UI

To open the MLflow UI, navigate to the directory containing your tracking database and run:

```bash
mlflow ui
```

The UI will be available at `http://127.0.0.1:5000` by default.

## API Reference

### `MLFlowConfig`

#### `MLFlowConfig.from_json(path: str | Path) -> MLFlowConfig`

Load config from a JSON file. Only keys matching `MLFlowConfig.__init__` parameters are used; extra keys are ignored.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str \| Path` | — | Path to the JSON config file |

---

#### `config.apply(run_name: str | None = None) -> MlflowClient`

Set the MLflow tracking URI, create the experiment if it doesn't exist, and start a new run. Returns an `MlflowClient` to pass to `MLFlowTracker`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `run_name` | `str \| None` | `None` | Run name; falls back to `run_name` from the JSON config if not provided |

---

#### `config.end_run()`

End the currently active MLflow run.

---

### `MLFlowTracker`

#### `MLFlowTracker(client: MlflowClient, config: MLFlowConfig, min_or_max: "min" | "max" | None)`

Create a tracker. Pass `min_or_max=None` for evaluation runs where no model comparison is needed.

| Parameter | Type | Description |
|-----------|------|-------------|
| `client` | `MlflowClient` | Returned by `config.apply()` |
| `config` | `MLFlowConfig` | The loaded config object |
| `min_or_max` | `"min" \| "max" \| None` | Whether to keep models with the lowest or highest `best_val` |

---

#### `tracker.log_training(model: Model, model_name: str, number_of_models_to_track: int, step: int | None = None)`

Log a training step. If `model.best_val` beats the current best objective, saves the model file, metrics, CSV artifacts, and configs to MLflow. When the tracked model count exceeds `number_of_models_to_track`, the worst-performing model is evicted from disk and MLflow.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `Model` | — | Dataclass containing the model, metrics, and config paths |
| `model_name` | `str` | — | Base name; a timestamp and counter are appended (e.g. `Resnet_2026-04-07_12-00-00_1`) |
| `number_of_models_to_track` | `int` | — | Maximum number of models to keep simultaneously |
| `step` | `int \| None` | `None` | Training epoch or iteration number associated with this log |

---

#### `tracker.log_evaluation(model: Model, model_name: str)`

Log an evaluation run. Does not save a model file — records `model_name` as the `evaluated_model` MLflow parameter and logs metrics, CSV artifacts, and configs.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `Model` | — | Dataclass containing the loaded model, metrics, and config paths |
| `model_name` | `str` | — | Full name of the model that was evaluated (as saved during training) |

---

#### `tracker.load_model(model_name: str) -> tuple[nn.Module, list[Path]]`

Load a previously logged model by its full name and return it along with its associated config file paths.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | — | Full model name including timestamp and counter (e.g. `Resnet_2026-04-07_12-00-00_1`) |

Returns `(nn.Module, list[Path])` — the loaded PyTorch model and a sorted list of `.json` config paths from the model's artifact directory.

---

#### `tracker.log_config(config_path: str | Path, save_config_for_model: bool = True, save_as_parameters: bool = True, artifact_path: str = "")`

Log a JSON config file as an MLflow artifact and optionally as run parameters. Skips silently if the file does not exist.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config_path` | `str \| Path` | — | Path to the config file (must be JSON if `save_as_parameters=True`) |
| `save_config_for_model` | `bool` | `True` | If `True` and `artifact_path` is set, saves the artifact under that subfolder; otherwise saves to the run root |
| `save_as_parameters` | `bool` | `True` | If `True`, parses the JSON and logs each top-level key as an MLflow parameter (skips keys already logged) |
| `artifact_path` | `str` | `""` | Artifact subfolder path (typically the model name); required for `save_config_for_model=True` to take effect |

---

#### `tracker.log_dataset(path: str | Path)`

Log a dataset path as the `dataset_path` MLflow parameter. Skips if the path does not exist.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str \| Path` | — | Path to the dataset file or directory |

---

#### `tracker.log_metrics(metrics: dict, step: int | None = None, save_as_artifact: bool = True, model_id: str | None = None, artifact_path: str = "")`

Log scalar metrics to MLflow and optionally save them as a `metrics.json` artifact.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metrics` | `dict` | — | Metric names to numeric values, e.g. `{"accuracy": 0.92, "loss": 0.25}` |
| `step` | `int \| None` | `None` | Training step or epoch to associate with the metrics |
| `save_as_artifact` | `bool` | `True` | If `True` and `artifact_path` is set, also saves a `metrics.json` artifact under `artifact_path` |
| `model_id` | `str \| None` | `None` | MLflow model ID to associate the metrics with |
| `artifact_path` | `str` | `""` | Artifact subfolder for the metrics JSON; required for artifact saving to take effect |

---

#### `tracker.log_metrics_artifact(metrics: dict, save_metrics_for_model: bool = True, artifact_path: str = "")`

Log structured (non-scalar) metrics as CSV artifacts. Each key in `metrics` becomes a separate `.csv` file.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metrics` | `dict` | — | Mapping of artifact name to a 2D structure: `list[list]`, `list[dict]`, `dict[str, list]`, or `np.ndarray`. Each value is saved as `<name>.csv`. |
| `save_metrics_for_model` | `bool` | `True` | If `True` and `artifact_path` is set, saves CSVs under that subfolder; otherwise saves to the run root |
| `artifact_path` | `str` | `""` | Artifact subfolder (typically the model name); required for `save_metrics_for_model=True` to take effect |

---

### `Model` dataclass

| Field | Type | Description |
|-------|------|-------------|
| `model` | `nn.Module` | The PyTorch model |
| `metrics` | `dict` | Scalar metrics logged to MLflow, e.g. `{"accuracy": 0.9, "loss": 0.3}` |
| `metrics_art` | `dict` | Non-scalar metrics saved as CSV artifacts. Each key becomes a separate `.csv` file. Supported value types: `list[list]` (e.g. a confusion matrix from `sklearn`), `list[dict]` (records with named columns), `dict[str, list]` (column-oriented), or `np.ndarray`. Values must be 2D. |
| `configs` | `list[str \| Path]` | Paths to JSON config files to associate with this model |
| `best_val` | `float` | The value compared against the tracker's objective (`min_or_max`) to decide whether to log this model |

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
