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

class MLFlowTracker:

    def __init__(self, client: MlflowClient, config: MLFlowConfig) -> None:
        self.artifact_dir = config.artifact_dir
        self.model_dir = config.models_dir
        self.client = client
        self.model_id = None
        self.model_number = 1

    def log_config(self, path: str | Path, save_as_parameters: bool = True):
        """
        Log a configuration file to MLflow as both an artifact and optional parameters.

        Parameters
        ----------
        path : str | Path
            Path to the configuration file (expected to be JSON if parameter logging is enabled).

        save_as_parameters : bool, default=True
            If True, attempts to parse the file as JSON and log its contents
            as MLflow parameters.
        """

        path = Path(path)
        data = None

        if not path.exists():
            print(f"[WARN] No such file: {path}. Skipping logging.")
            return

        if save_as_parameters:
            try:
                with path.open("r") as f:
                    data = json.load(f)

                if not isinstance(data, dict):
                    print(f"[WARN] Config is not a dict. Skipping param logging.")
                    data = None

            except Exception as e:
                print(f"[WARN] Failed to read config {path}: {e}")
                data = None

            if data:
                try:
                    run = mlflow.active_run()
                    if run:
                        client = mlflow.MlflowClient()
                        existing_keys = set(
                            client.get_run(run.info.run_id).data.params.keys()
                        )
                    else:
                        existing_keys = set()

                    new_params = {
                        k: v for k, v in data.items()
                        if k not in existing_keys
                    }

                    if new_params:
                        mlflow.log_params(new_params)

                except Exception as e:
                    print(f"[WARN] Failed to log params: {e}")

        mlflow.log_artifact(str(path))
        print(f"COnfig from path: {path} has been logged to MLFlow.")

    def log_dataset(self, path: str | Path):
        """Log dataset path to MLFlow"""
        try:
            dataset_path = Path(path)
            mlflow.log_param("dataset_path", str(dataset_path))
            print(f"Dataset from path: {path} has been logged to MLFlow")
        except FileNotFoundError:
            print(f"No such file or directory: {path}. Skipping this dataset logging.")

    def log_metrics(self, metrics: dict, step: int | None = None):
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
            mlflow.log_metrics(metrics, step=step)
            print(f"Metrics: {metrics} have been logged to MLFlow ")  
        except Exception as e:
            print(f"[WARN] Failed to log metrics: {e}")

    def log_metrics_artifact(self, metrics: dict):
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
                        mlflow.log_artifact(str(path))
                        print(f"Metric: {name} file has been logged to MLFlow artifacts.")
                    except Exception as e:
                        print(f"[WARN] Failed for {name}: {e}")
        except Exception as e:
            print(f"[WARN] Artifact logging failed entirely: {e}")
            
    def log_model(self, model, model_name: str):
        """    
        Log a PyTorch model to MLflow, replacing any previously logged model.

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
        
        model_name = f"{model_name}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{self.model_number}"
        model_info = mlflow.pytorch.log_model(pytorch_model=model, name=model_name) # type: ignore
        self.model_number += 1

        self.model_id = model_info.model_id
        print(f"Model {self.model_id} logged to MLFlow")

        # model extension change from .pth to .pt
        pth_path = os.path.join(self.artifact_dir, self.model_dir, self.model_id, "artifacts/data/model.pth")
        dir_path = os.path.dirname(pth_path)
        pt_path = os.path.join(dir_path, model_name + ".pt")
        os.rename(pth_path, pt_path)