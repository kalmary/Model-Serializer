import logging
import mlflow
import mlflow.pytorch
import json
import shutil
from pathlib import Path
from mlflow.tracking import MlflowClient
from src.config import MLFlowConfig
import pandas as pd
import tempfile
from datetime import datetime
from typing import Literal
from dataclasses import dataclass
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

@dataclass
class Model:
     model: nn.Module
     metrics: dict
     metrics_art: dict
     configs: list[str | Path]
     best_val: float


class MLFlowTracker:

    def __init__(self, client: MlflowClient, config: MLFlowConfig, min_or_max: Literal["min", "max"] | None) -> None:
        self.artifact_dir = config.artifact_dir
        self.model_dir = config.models_dir
        self.client = client
        self.run_id = mlflow.active_run().info.run_id   # type: ignore
        self.model_id = None
        self.model_name: str = ""
        self.model_number: int = 1
        self.dropped_model_id = None
        self.models_id_list: list[tuple[str | None, float]] = []

        self.min_or_max = min_or_max
        if self.min_or_max == "max":
            self.best_objective: float = float('-inf')
        else:
            self.best_objective: float = float('inf')



    def log_config(self, config_path: str | Path, save_config_for_model: bool = True, save_as_parameters: bool = True, artifact_path: str = ""):
        """
        Log a configuration file to MLflow as both an artifact and optional parameters.

        Parameters
        ----------
        path : str | Path
            Path to the configuration file (expected to be JSON if parameter logging is enabled).
        save_config_for_model: bool, default = True
            If True saves config artifact to folder with the model name. Else it saves to run global artifacts.
        save_as_parameters : bool, default=True
            If True, attempts to parse the file as JSON and log its contents
            as MLflow parameters.
        """

        config_path = Path(config_path)
        data = None

        if not config_path.exists():
            logger.warning(f"No such file: {config_path}. Skipping logging.")
            return

        if save_as_parameters:
            try:
                with config_path.open("r") as f:
                    data = json.load(f)

                if not isinstance(data, dict):
                    logger.warning("Config is not a dict. Skipping param logging.")
                    data = None

            except Exception as e:
                logger.warning(f"Failed to read config {config_path}: {e}")
                data = None

            if data:
                try:
                    existing_keys = set(
                        self.client.get_run(self.run_id).data.params.keys()
                    )
                    new_params = {k: v for k, v in data.items() if k not in existing_keys}
                    if new_params:
                        mlflow.log_params(new_params)

                except Exception as e:
                    logger.warning(f"Failed to log params: {e}")
        if save_config_for_model and artifact_path != "":
            mlflow.log_artifact(local_path=str(config_path), artifact_path=artifact_path)
            logger.info(f"Model config from path: {config_path} has been logged to MLFlow.")
        else:
            mlflow.log_artifact(local_path=str(config_path))
            logger.info(f"Global config from path: {config_path} has been logged to MLFlow.")

    def log_dataset(self, path: str | Path):
        """Log dataset path to MLFlow"""
        dataset_path = Path(path)
        if not dataset_path.exists():
            logger.warning(f"No such file or directory: {path}. Skipping this dataset logging.")
            return
        mlflow.log_param("dataset_path", str(dataset_path))
        logger.info(f"Dataset from path: {path} has been logged to MLFlow")

    def log_metrics(self, metrics: dict, step: int | None = None, save_as_artifact: bool = True, model_id: str | None = None, artifact_path: str = ""):
        """    
        Log scalar metrics to MLflow.

        Parameters
        ----------
        metrics : dict
            Dictionary of metric names to numeric values (int or float).
            Example: {"loss": 0.25, "accuracy": 0.92}

        step : int | None, optional
            Step index to associate with the metrics (e.g., training iteration or epoch).
        I   f None, MLflow logs metrics without a step.
        """

        try:
            mlflow.log_metrics(metrics, step=step, model_id=model_id)
            logger.info(f"Metrics: {metrics} have been logged to MLFlow")

            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "metrics.json"
                with path.open("w") as f:
                    json.dump(metrics, f)
                if save_as_artifact and artifact_path != "":
                    mlflow.log_artifact(local_path=str(path), artifact_path=artifact_path)

                logger.info(f"Metrics: {metrics} file has been logged to MLFlow artifacts.")
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")

    def log_metrics_artifact(self, metrics: dict, save_metrics_for_model: bool = True, artifact_path: str = ""):
        """    
        Log structured metrics as CSV artifacts in MLflow.

        This is intended for non-scalar outputs such as confusion matrices,
        per-class metrics, or tabular evaluation results.

        Parameters
        ----------
        metrics : dict
            Dictionary mapping artifact names to data structures convertible
            to pandas DataFrames.
            Example:
                {
                    "confusion_matrix": [[50, 2], [1, 47]],
                    "per_class_metrics": [{"class": 0, "f1": 0.91}, ...]
                }
        """
        try:
            with tempfile.TemporaryDirectory() as tmp:
                for name, data in metrics.items():
                    try:
                        path = Path(tmp) / f"{name}.csv"
                        pd.DataFrame(data).to_csv(path, index=False)
                        if save_metrics_for_model and artifact_path != "":
                            mlflow.log_artifact(local_path=str(path), artifact_path=artifact_path)
                        else:
                            mlflow.log_artifact(str(path))
                        logger.info(f"Metric: {name} file has been logged to MLFlow artifacts.")
                    except Exception as e:
                        logger.warning(f"Failed for {name}: {e}")
        except Exception as e:
            logger.warning(f"Artifact logging failed entirely: {e}")
            
    def log_model(self, model: nn.Module, model_name: str, mode: Literal["training", "evaluation"], number_of_models_to_track: int = 1, step: int | None = None):
        self.model_name = f"{model_name}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{self.model_number}"
        model_info = mlflow.pytorch.log_model(pytorch_model=model, name=self.model_name, step=step) # type: ignore
        self.model_number += 1
        self.model_id = model_info.model_id
        
        # model extension change from .pth to .pt
        pth_path = Path(self.artifact_dir) / self.model_dir / self.model_id / "artifacts/data/model.pth"
        pt_path = pth_path.with_name(self.model_name + ".pt")
        pth_path.rename(pt_path)
        if mode == "training":
            self.update_models_tracked(number_of_models_to_track=number_of_models_to_track)
            if self.dropped_model_id is not None:
                self.del_model_folder(self.dropped_model_id)
                self.del_model_art(self.dropped_model_id)
                self.client.delete_logged_model(self.dropped_model_id)
                self.dropped_model_id = None

    def check_if_better(self, objective:float) -> bool:

        if self.min_or_max == "min" and objective < self.best_objective:
                self.best_objective = objective
                return True
        elif self.min_or_max == "max" and objective > self.best_objective:
                self.best_objective = objective
                return True
        return False
    
    def update_models_tracked(self, number_of_models_to_track: int):
        self.dropped_model_id = None
        if self.model_id is not None:
            self.models_id_list.append((self.model_id, self._current_best_val))
        if len(self.models_id_list) > number_of_models_to_track:
            if self.min_or_max == "max":
                worst = min(self.models_id_list, key=lambda x: x[1])
            else:
                worst = max(self.models_id_list, key=lambda x: x[1])
            self.dropped_model_id = worst[0]
            self.models_id_list.remove(worst)

    
    def del_model_folder(self, dropped_model_id):
        dropped_model_path = Path(self.artifact_dir) / self.model_dir / dropped_model_id

        if dropped_model_path.exists():
            shutil.rmtree(dropped_model_path)

    def del_model_art(self, dropped_model_id):

        model_name = mlflow.get_logged_model(dropped_model_id).name
        dropped_model_art_path = Path(self.artifact_dir) / self.run_id / "artifacts" / model_name
        logger.info(f"Deleting artifact folder: {dropped_model_art_path}")
        if dropped_model_art_path.exists():
            shutil.rmtree(dropped_model_art_path)


    def log_training(self, model: Model, model_name:str, number_of_models_to_track: int, step: int | None = None):

        if self.check_if_better(objective=model.best_val):
            self._current_best_val = model.best_val
            self.log_model(model=model.model, mode="training", model_name=model_name, number_of_models_to_track=number_of_models_to_track, step=step or 0)
            self.log_metrics(metrics=model.metrics, step=step, model_id=self.model_id, artifact_path=self.model_name)
            self.log_metrics_artifact(metrics=model.metrics_art, save_metrics_for_model=True, artifact_path=self.model_name)
            for config_path in model.configs:
                self.log_config(config_path=config_path, save_config_for_model=True, save_as_parameters=True, artifact_path=self.model_name)

    def log_evaluation(self, model: Model, model_name:str):
        self.model_name = model_name
        mlflow.log_param("evaluated_model", model_name)
        self.log_metrics(metrics=model.metrics, artifact_path=self.model_name)
        self.log_metrics_artifact(metrics=model.metrics_art, save_metrics_for_model=True, artifact_path=self.model_name)
        for config_path in model.configs:
            self.log_config(config_path=config_path, save_config_for_model=True, save_as_parameters=True, artifact_path=self.model_name)

    
    def load_model(self, model_name: str) -> tuple[nn.Module, list[Path]]:
        """
        Load a previously logged model and its config files by name.

        Parameters
        ----------
        model_name : str
            The full name of the model (e.g., "Resnet_2026-03-25_13-30-40_5").

        Returns
        -------
        tuple[nn.Module, list[Path]]
            The loaded PyTorch model and list of associated config file paths.
        """
        active_run = mlflow.active_run()
        if active_run is None:
            raise RuntimeError("No active MLflow run. Call config.apply() first.")
        experiment_id = active_run.info.experiment_id

        results = self.client.search_logged_models(
            experiment_ids=[experiment_id],
            filter_string=f"name = '{model_name}'"
        )
        
        if not results:
            raise ValueError(f"No model found with name '{model_name}'")

        logged_model = results[0]
        model_id = logged_model.model_id
        source_run_id = logged_model.source_run_id
        if source_run_id is None:
            raise ValueError(f"Model '{model_name}' has no source run ID")

        # Load the .pt file from the model directory
        model_path = Path(self.artifact_dir) / self.model_dir / model_id / "artifacts" / "data" / f"{model_name}.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        loaded_model = torch.load(model_path, weights_only=False)
        logger.info(f"Loaded model from {model_path}")

        # Find config files from the source run's artifact directory
        config_dir = Path(self.artifact_dir) / source_run_id / "artifacts" / model_name
        config_paths = sorted(config_dir.glob("*.json")) if config_dir.exists() else []
        logger.info(f"Found {len(config_paths)} config files for model '{model_name}'")

        return loaded_model, config_paths
