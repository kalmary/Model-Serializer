import torch
import mlflow
from .dataset import TitanicDataset
from .model import TitanicModel
from .train import train_model
from .evaluate import evaluate
from .utils import load_json
from src.ml_flow_utils.config import MLFlowConfig
from src.ml_flow_utils.tracker import MLFlowTracker


dataset_path = "/home/michal/code/Model-Serializer/src/tests/titanic_pytorch_mlflow_test/data/Titanic-Dataset.csv"
model_config_path = "/home/michal/code/Model-Serializer/src/tests/titanic_pytorch_mlflow_test/configs/model_config.json"
train_config_path = "/home/michal/code/Model-Serializer/src/tests/titanic_pytorch_mlflow_test/configs/train_config.json"

def main():

    config = MLFlowConfig.from_json('/home/michal/code/Model-Serializer/config/mlflow_config.json')
    client = config.apply()

    logger = MLFlowTracker(client=client, config=config)

    model_config = load_json(model_config_path)
    train_config = load_json(train_config_path)

    dataset = TitanicDataset(dataset_path)

    model = TitanicModel(model_config)

    model, test_loader = train_model(dataset, model, train_config, logger)

    
    logger.log_model_config("/home/michal/code/Model-Serializer/config/config_model_randlanet.json")
    logger.log_train_config("/home/michal/code/Model-Serializer/config/config_train_randlanet.json")
    logger.log_dataset(dataset_path)
    logger.log_model(model=model, model_name="model_test")
    
    test_accuracy, cm, rc = evaluate(model, test_loader)

    logger.log_metrics({"test_acc": test_accuracy})
    logger.log_metrics_artfact({"confusion_matrix": cm, "roc_curve": rc})

    print("Test Accuracy:", test_accuracy)

    #torch.save(model.state_dict(), "model.pt")


if __name__ == "__main__":
    main()