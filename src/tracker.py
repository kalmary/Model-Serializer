import mlflow
import mlflow.pytorch
import json
import os
import shutil
from pathlib import Path
from mlflow.tracking import MlflowClient
from src.config import MLFlowConfig
import pandas as pd
import tempfile
from datetime import datetime
from typing import Literal
from dataclasses import dataclass
from collections import deque
import torch.nn as nn

@dataclass
class Model:
     model: nn.Module
     metrics: dict
     metrics_art: dict
     config: str | Path
     best_val: float


class MLFlowTracker:

    def __init__(self, client: MlflowClient, config: MLFlowConfig, number_of_models_to_track: int, min_or_max: Literal["min", "max"]) -> None:
        self.artifact_dir = config.artifact_dir
        self.model_dir = config.models_dir
        self.client = client
        self.run_id = mlflow.active_run().info.run_id   # type: ignore
        self.model_id = None
        self.model_name = ""
        self.model_number = 1
        self.dropped_model_id = None
        self.number_of_models_to_track = number_of_models_to_track
        self.min_or_max = min_or_max
        self.models_id_list = deque(maxlen=self.number_of_models_to_track)
        if self.min_or_max == "max":
            self.best_objective: float = float('-inf')
        else:
            self.best_objective: float = float('inf')



    def log_config(self, config_path: str | Path, save_config_for_model: bool = True, save_as_parameters: bool = True):
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
            print(f"[WARN] No such file: {config_path}. Skipping logging.")
            return

        if save_as_parameters:
            try:
                with config_path.open("r") as f:
                    data = json.load(f)

                if not isinstance(data, dict):
                    print("[WARN] Config is not a dict. Skipping param logging.")
                    data = None

            except Exception as e:
                print(f"[WARN] Failed to read config {config_path}: {e}")
                data = None

            if data:
                try:
                    existing_keys = set(
                        self.client.get_run(self.run_id).data.params.keys()
                    )

                    new_params = {
                        k: v for k, v in data.items()
                        if k not in existing_keys
                    }

                    if new_params:
                        mlflow.log_params(new_params)

                except Exception as e:
                    print(f"[WARN] Failed to log params: {e}")
        if save_config_for_model and self.model_name != "":
            mlflow.log_artifact(local_path=str(config_path), artifact_path=self.model_name)
            print(f"Model config from path: {config_path} has been logged to MLFlow.")            
        else:
            mlflow.log_artifact(local_path=str(config_path))
            print(f"Global config from path: {config_path} has been logged to MLFlow.")

    def log_dataset(self, path: str | Path):
        """Log dataset path to MLFlow"""
        try:
            dataset_path = Path(path)
            mlflow.log_param("dataset_path", str(dataset_path))
            print(f"Dataset from path: {path} has been logged to MLFlow")
        except FileNotFoundError:
            print(f"No such file or directory: {path}. Skipping this dataset logging.")

    def log_metrics(self, metrics: dict, step: int | None = None, save_as_artifact: bool = True, model_id: str | None = None):
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
            print(f"Metrics: {metrics} have been logged to MLFlow ")

            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / f"metrics_step_{step}.json"
                with path.open("w") as f:
                    json.dump(metrics, f)
                if save_as_artifact and self.model_name != "":
                    mlflow.log_artifact(local_path=str(path), artifact_path=self.model_name)

                print(f"Metrics: {metrics} file has been logged to MLFlow artifacts.")
        except Exception as e:
            print(f"[WARN] Failed to log metrics: {e}")

    def log_metrics_artifact(self, metrics: dict, save_metrics_for_model: bool = True):
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
                        if save_metrics_for_model and self.model_name != "":
                            mlflow.log_artifact(local_path=str(path), artifact_path=self.model_name)
                        else:
                            mlflow.log_artifact(str(path))
                        print(f"Metric: {name} file has been logged to MLFlow artifacts.")
                    except Exception as e:
                        print(f"[WARN] Failed for {name}: {e}")
        except Exception as e:
            print(f"[WARN] Artifact logging failed entirely: {e}")
            
    def log_model(self, model, model_name: str):
        #curently not used
        """    
        Log a single PyTorch model to MLflow, replacing any previously logged model.

        Parameters
        ----------
        model : torch.nn.Module
            The PyTorch model instance to be logged.

        model_name : str
            Name under which the model will be registered in MLflow.
        """
        if self.model_id is not None:
            model_path = os.path.join(self.artifact_dir, self.model_dir, self.model_id)

            if os.path.exists(model_path):
                shutil.rmtree(model_path)

            self.client.delete_logged_model(self.model_id)
            print(f"Model: {self.model_id} has been removed from MLFlow")
        
        self.model_name = f"{model_name}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{self.model_number}"
        model_info = mlflow.pytorch.log_model(pytorch_model=model, name=self.model_name) # type: ignore
        self.model_number += 1

        self.model_id = model_info.model_id
        print(f"Model {self.model_id} logged to MLFlow")

        # model extension change from .pth to .pt
        pth_path = os.path.join(self.artifact_dir, self.model_dir, self.model_id, "artifacts/data/model.pth")
        dir_path = os.path.dirname(pth_path)
        pt_path = os.path.join(dir_path, self.model_name + ".pt")
        os.rename(pth_path, pt_path)

    def log_models(self, model, model_name: str, step: int = 0):
        #curenlty used
        self.model_name = f"{model_name}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{self.model_number}"
        model_info = mlflow.pytorch.log_model(pytorch_model=model, name=self.model_name, step=step) # type: ignore
        self.model_number += 1
        self.model_id = model_info.model_id
        
        # model extension change from .pth to .pt
        pth_path = os.path.join(self.artifact_dir, self.model_dir, self.model_id, "artifacts/data/model.pth")
        dir_path = os.path.dirname(pth_path)
        pt_path = os.path.join(dir_path, self.model_name + ".pt")
        os.rename(pth_path, pt_path)

        self.update_models_tracked()
        if self.dropped_model_id is not None:
            self.del_model_folder(self.dropped_model_id)
            self.del_model_art(self.dropped_model_id)
            self.client.delete_logged_model(self.dropped_model_id)

    def check_if_better(self, objective:float) -> bool:

        if self.min_or_max == "min" and objective < self.best_objective:
                self.best_objective = objective
                return True
        elif self.min_or_max == "max" and objective > self.best_objective:
                self.best_objective = objective
                return True
        return False
    
    def update_models_tracked(self):

        if len (self.models_id_list) == self.number_of_models_to_track:
            self.dropped_model_id = self.models_id_list[0]
        self.models_id_list.append(self.model_id)  

    
    def del_model_folder(self, dropped_model_id):
        dropped_model_path = os.path.join(self.artifact_dir, self.model_dir, dropped_model_id)

        if os.path.exists(dropped_model_path):
            shutil.rmtree(dropped_model_path)    

    def del_model_art(self, dropped_model_id):

        model_name = mlflow.get_logged_model(dropped_model_id).name
        dropped_model_art_path = os.path.join(
            self.artifact_dir,
            self.run_id,
            "artifacts",
            model_name
        )
        print(f"Deleting artifact folder: {dropped_model_art_path}")
        if os.path.exists(dropped_model_art_path):
            shutil.rmtree(dropped_model_art_path)


    def log_training(self, model: Model, model_name:str, step: int | None = None):

        if self.check_if_better(objective=model.best_val):
            self.log_models(model=model.model, model_name=model_name, step=step or 0)
            self.log_metrics(metrics=model.metrics, step=step, model_id=self.model_id)
            self.log_metrics_artifact(metrics=model.metrics_art,save_metrics_for_model=True)
            self.log_config(config_path=model.config, save_config_for_model=True, save_as_parameters=False)
            