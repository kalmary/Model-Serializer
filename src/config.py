import json
import inspect
import logging
from pathlib import Path
import mlflow
from mlflow import MlflowClient

logger = logging.getLogger(__name__)

class MLFlowConfig:
    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
        run_name: str,
        artifact_dir: str,
        run_tags: dict | None = None,
        experiment_tags: dict | None = None,
        models_dir: str = "models"
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.experiment_tags = experiment_tags
        self.run_name = run_name
        self.run_tags = run_tags    
        self.artifact_dir = artifact_dir
        self.models_dir = models_dir

    @classmethod
    def from_json(cls, path: str | Path):
        """Create MLFlowConfig from a JSON file."""
        path = Path(path)

        with path.open("r") as f:
            data = json.load(f)

        valid_params = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}
        filtered = {k: v for k, v in data.items() if k in valid_params}
        return cls(**filtered)

    def apply(self, run_name: str | None = None) -> MlflowClient:
        """
        Configure MLflow environment and ensure the experiment exists.
        """
        if run_name is None:
            run_name = self.run_name

        mlflow.set_tracking_uri(self.tracking_uri)

        experiment = mlflow.get_experiment_by_name(self.experiment_name)

        if experiment is None:
            self.experiment_id = mlflow.create_experiment(
                name=self.experiment_name,
                artifact_location=self.artifact_dir,
                tags=self.experiment_tags)
        else:
            self.experiment_id = experiment.experiment_id

        mlflow.set_experiment(self.experiment_name)

        try:
            mlflow.start_run(run_name=run_name, tags=self.run_tags)
        except Exception:
            logger.warning("Another run is already active. To start a new run, first end this one")

        return MlflowClient(tracking_uri=self.tracking_uri)

    def end_run(self):
        """End the current MLflow run."""
        mlflow.end_run()
