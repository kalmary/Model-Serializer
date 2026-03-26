MLflow Config JSON Reference
============================

Fields used by MLFlowConfig (src/config.py):

  tracking_uri       - URI of the MLflow tracking server (e.g. sqlite:///path/to/mlflow.db)
  experiment_name    - Name of the MLflow experiment
  run_name           - Default run name (can be overridden in config.apply())
  artifact_dir       - Base directory where all artifacts (models, metrics, configs) are stored
  run_tags           - (optional) Dict of metadata tags attached to runs, used for filtering in UI
  experiment_tags    - (optional) Dict of metadata tags attached to the experiment
  models_dir         - (optional, default "models") Subdirectory name for model storage within artifact_dir

Additional fields in config JSON (not loaded by MLFlowConfig, but useful for project conventions):

  registry_uri       - Location of the model registry backend (typically same as tracking_uri)
  config_dir         - Directory containing config files
  plot_dir           - Directory for visualization outputs

Example (src/tests/example/mlflow_config.json):

  {
    "tracking_uri": "sqlite:////path/to/mlflow.db",
    "artifact_dir": "/path/to/artifacts",
    "experiment_name": "my_experiment",
    "run_name": "my_run",
    "experiment_tags": {"framework": "pytorch"},
    "run_tags": {"model_type": "ResNet"},
    "model_dir": "model"
  }
