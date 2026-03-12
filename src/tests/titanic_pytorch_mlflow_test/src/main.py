import torch

from src.dataset import TitanicDataset
from src.model import TitanicModel
from src.train import train_model
from src.evaluate import evaluate
from src.utils import load_json


def main():

    model_config = load_json("configs/model_config.json")
    train_config = load_json("configs/train_config.json")

    dataset = TitanicDataset("data/Titanic-Dataset.csv")

    model = TitanicModel(model_config)

    model, test_loader = train_model(dataset, model, train_config)

    accuracy = evaluate(model, test_loader)

    print("Test Accuracy:", accuracy)

    torch.save(model.state_dict(), "model.pt")


if __name__ == "__main__":
    main()