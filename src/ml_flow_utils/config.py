import json
from pathlib import Path
from datetime import datetime
import mlflow
import inspect

class MLFlowConfig:
    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
        artifact_dir: str,
        #run_name_template: str | None = None,
        tags: dict | None = None,
        register_model: bool = False,
        registered_model_name: str | None = None,
        registry_stage: str | None = None
    ):
        self.tracking_uri = tracking_uri
        self.artifact_dir = artifact_dir
        self.experiment_name = experiment_name
        #self.run_name_template = run_name_template
        self.tags = tags or {}

        self.register_model = register_model
        self.registered_model_name = registered_model_name
        self.registry_stage = registry_stage

    @classmethod
    def from_json(cls, path: str | Path):
        """Create MLFlowConfig from a JSON file."""
        path = Path(path)

        with path.open("r") as f:
            data = json.load(f)

        valid_params = cls.__init__.__code__.co_varnames[:cls.__init__.__code__.co_argcount]
        filtered = {k: v for k, v in data.items() if k in valid_params}
        return cls(**filtered)

    def apply(self):
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

    '''
    def generate_run_name(self, context: dict | None = None):
        """
        Generate a run name using the run_name_template.
        """
        if not self.run_name_template:
            return None

        context = context or {}

        context.setdefault(
            "timestamp",
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )

        return self.run_name_template.format(**context)
    '''