import mlflow
import json
from pathlib import Path
import mlflow.pytorch

class MLFlowTracker:

    def __init__(self) -> None:
        pass

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

    def log_model(self, model, artifact_path: str = "model"):
        mlflow.pytorch.log_model(model, artifact_path=artifact_path)

    def log_metrics(self):

        mlflow.log_metrics
        pass


    '''
    def start_tracking(self, train_param_path, model_arch_path, dataset_path):
    with mlflow.start_run():

        self.train_param_tracker(train_param_path)
        self.model_arch_tracker(model_arch_path)
        self.dataset_tracker(dataset_path)  
    '''





# add exceptions (like wrong path)
#change var names