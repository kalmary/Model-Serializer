import mlflow
import mlflow.pytorch
import json
import os
import shutil
from pathlib import Path
from mlflow.tracking import MlflowClient
from src.ml_flow_utils.config import MLFlowConfig
import pandas as pd
import tempfile

class MLFlowTracker:

    def __init__(self, client: MlflowClient, artifact_dir: str) -> None:
        self.artifact_dir = artifact_dir
        self.client = client
        self.model_id = None

    def log_train_config(self, path: str | Path, save_as_parameters: bool = True):
        path = Path(path)

        if save_as_parameters:
            with path.open("r") as f:
                data = json.load(f)

            run = mlflow.active_run()
            if run:
                existing_keys = set(mlflow.MlflowClient().get_run(run.info.run_id).data.params.keys())
            else:
                existing_keys = set()

            new_params = {k: v for k, v in data.items() if k not in existing_keys}

            if new_params:
                mlflow.log_params(new_params)

        mlflow.log_artifact(str(path))

    def log_model_config(self, path: str | Path, save_as_parameters: bool = True):
        path = Path(path)

        if save_as_parameters:
            with path.open("r") as f:
                data = json.load(f)

            run = mlflow.active_run()
            if run:
                existing_keys = set(mlflow.MlflowClient().get_run(run.info.run_id).data.params.keys())
            else:
                existing_keys = set()

            new_params = {k: v for k, v in data.items() if k not in existing_keys}

            if new_params:
                mlflow.log_params(new_params)

        mlflow.log_artifact(str(path))

    def log_dataset(self, path: str | Path):
        dataset_path = Path(path)
        mlflow.log_param("dataset_path", str(dataset_path))

    def log_metrics(self, metrics: dict, step: int | None = None):
        mlflow.log_metrics(metrics, step=step)

    def log_metrics_artfact(self, metrics: dict):

        with tempfile.TemporaryDirectory() as tmp:
            for name, data in metrics.items():
                path = Path(tmp) / f"{name}.csv"
                pd.DataFrame(data).to_csv(path, index=False)
                mlflow.log_artifact(str(path))
            
    def log_model(self, model, model_name: str):

        if self.model_id is not None:
            model_path = os.path.join(self.artifact_dir, "models", self.model_id)

            if os.path.exists(model_path):
                shutil.rmtree(model_path)

            self.client.delete_logged_model(self.model_id)
        
        model_info = mlflow.pytorch.log_model(pytorch_model=model, name=model_name) # type: ignore
        self.model_id = model_info.model_id
        self.model_key = model_info.tags

        # model extension change from .pth to .pt
        pth_path = os.path.join(self.artifact_dir, "models", self.model_id, "artifacts/data/model.pth")
        pt_path = os.path.splitext(pth_path)[0] + ".pt"
        os.rename(pth_path, pt_path)


# add exceptions (like wrong path)