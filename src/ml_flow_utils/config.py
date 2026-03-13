import json
from pathlib import Path
from datetime import datetime
import mlflow
from mlflow import MlflowClient

class MLFlowConfig:
    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
        artifact_dir: str,
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.artifact_dir = artifact_dir

    @classmethod
    def from_json(cls, path: str | Path):
        """Create MLFlowConfig from a JSON file."""
        path = Path(path)

        with path.open("r") as f:
            data = json.load(f)

        valid_params = cls.__init__.__code__.co_varnames[:cls.__init__.__code__.co_argcount]
        filtered = {k: v for k, v in data.items() if k in valid_params}
        return cls(**filtered)

    def apply(self) -> MlflowClient:
        """
        Configure MLflow environment and ensure the experiment exists.
        """
        mlflow.set_tracking_uri(self.tracking_uri)

        experiment = mlflow.get_experiment_by_name(self.experiment_name)

        if experiment is None:
            self.experiment_id = mlflow.create_experiment(self.experiment_name, self.artifact_dir)
        else:
            self.experiment_id = experiment.experiment_id

        mlflow.set_experiment(self.experiment_name)

        return MlflowClient(tracking_uri=self.tracking_uri)
