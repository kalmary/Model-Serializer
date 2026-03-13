import torch
import mlflow
from .dataset import TitanicDataset
from .model import TitanicModel
from .train import train_model
from .evaluate import evaluate
from .utils import load_json
from src.ml_flow_utils.config import MLFlowConfig
from src.ml_flow_utils.tracker import MLFlowTracker


def main():

    config = MLFlowConfig.from_json('/home/michal/code/Model-Serializer/config/mlflow_config.json')
    client = config.apply()

    with mlflow.start_run():

        model_config = load_json("/home/michal/code/Model-Serializer/src/tests/titanic_pytorch_mlflow_test/configs/model_config.json")
        train_config = load_json("/home/michal/code/Model-Serializer/src/tests/titanic_pytorch_mlflow_test/configs/train_config.json")

        dataset = TitanicDataset("/home/michal/code/Model-Serializer/src/tests/titanic_pytorch_mlflow_test/data/Titanic-Dataset.csv")

        model = TitanicModel(model_config)

        model, test_loader = train_model(dataset, model, train_config)

        logger = MLFlowTracker(client, config.artifact_dir)
        logger.log_model(model=model, model_name="test1")

        input()

        logger.log_model(model=model, model_name="test1")
        accuracy = evaluate(model, test_loader)

        print("Test Accuracy:", accuracy)

        #torch.save(model.state_dict(), "model.pt")


if __name__ == "__main__":
    main()