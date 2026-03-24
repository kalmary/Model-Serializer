# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Model-Serializer is a PyTorch model tracking and management library built on MLflow. It tracks the top N best-performing models during training, automatically evicting older models when the limit is exceeded.

## Architecture

Two main classes in `src/`:

- **`MLFlowConfig`** (`src/config.py`): Loads MLflow configuration from JSON, sets up tracking URI, experiment, and starts a run via `apply()`. Returns an `MlflowClient`.
- **`MLFlowTracker`** (`src/tracker.py`): Core tracking class. Maintains a bounded deque of model IDs (`number_of_models_to_track`). On each `log_training()` call, checks if the new model beats the current best (min or max objective), and if so: logs the model, metrics, metric artifacts (CSVs), and config. Automatically deletes the oldest tracked model (filesystem + MLflow artifacts) when the queue is full.
- **`Model`** (`src/tracker.py`): Dataclass bundling a `nn.Module`, scalar metrics dict, non-scalar metrics dict (logged as CSV artifacts), config path, and `best_val` for comparison.

Flow: `MLFlowConfig.from_json()` -> `config.apply()` -> `MLFlowTracker(client, config, ...)` -> loop over `tracker.log_training(model, name, step)`.

## Running Tests

No formal test framework. Run the manual test script from the repo root:

```
python -m src.tests.michal_fake_env.main
```

This creates 7 fake models with varying accuracy and tests the tracker with `number_of_models_to_track=2` using max objective.

## Dependencies

Python 3.13, managed with `uv`. Key packages: `mlflow`, `torch`, `pandas`, `numpy`.

## Conventions

- All imports use absolute paths from repo root: `from src.config import MLFlowConfig`
- Run all commands from the repo root directory
- MLflow config is stored as JSON files (see `src/mlflow_config.json` for structure)
- Models are saved as `.pt` files (renamed from MLflow's default `.pth`)
