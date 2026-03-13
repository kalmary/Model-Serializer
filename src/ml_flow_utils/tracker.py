import mlflow
import mlflow.pytorch
import json
import os
import shutil
from pathlib import Path
from mlflow.tracking import MlflowClient
from src.ml_flow_utils.config import MLFlowConfig

class MLFlowTracker:

    def __init__(self, client: MlflowClient, artifact_dir: str) -> None:
        self.artifact_dir = artifact_dir
        self.client = client
        self.model_id = None

    def log_train_param(self, path: str | Path):
        path = Path(path)

        with path.open("r") as f:
            data = json.load(f)
        
        mlflow.log_params(data)
        mlflow.log_artifact(str(path))

    def log_model_arch(self, path: str | Path):
        mlflow.log_artifact(str(path))        

    def log_dataset(self, path: str | Path):
        dataset_path = Path(path)
        mlflow.log_param("dataset_name", dataset_path.name)
        mlflow.log_param("dataset_path", str(dataset_path))
        #change to multiple or one

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

        
        #change the "models" path


    def log_metrics(self):

        mlflow.log_metrics
        pass





# add exceptions (like wrong path)
#change var names