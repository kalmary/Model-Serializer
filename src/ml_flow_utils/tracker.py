import mlflow
import mlflow.pytorch
import json
import os
import shutil
from pathlib import Path
from mlflow.tracking import MlflowClient

class MLFlowTracker:

    def __init__(self, client: MlflowClient) -> None:
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
        artifact_dir = "/home/michal/code/Model-Serializer/src/tests/titanic_pytorch_mlflow_test/mlflow_data/artifacts"
        if self.model_id is not None:
            model_path = os.path.join(artifact_dir, "models", self.model_id)
            print(model_path)
            shutil.rmtree(model_path)
            self.client.delete_logged_model(self.model_id)
        
        model_info = mlflow.pytorch.log_model(pytorch_model=model, name=model_name) # type: ignore
        self.model_id = model_info.model_id
        self.model_key = model_info.tags



    def log_metrics(self):

        mlflow.log_metrics
        pass





# add exceptions (like wrong path)
#change var names