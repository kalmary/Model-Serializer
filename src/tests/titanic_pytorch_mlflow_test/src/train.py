import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from src.ml_flow_utils.tracker import MLFlowTracker

def train_model(dataset, model, train_config, logger):

    batch_size = train_config["batch_size"]
    lr = train_config["learning_rate"]
    epochs = train_config["epochs"]

    test_size = int(len(dataset) * train_config["test_split"])
    train_size = len(dataset) - test_size

    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):

        model.train()

        total_loss = 0
        correct = 0
        total = 0

        for X, y in train_loader:

            optimizer.zero_grad()

            preds = model(X)

            loss = criterion(preds, y)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            # Convert probabilities to predictions
            predicted = (preds > 0.5).float()

            correct += (predicted == y).sum().item()
            total += y.size(0)

        accuracy = correct / total
        if logger:
            logger.log_metrics({"accuracy": accuracy}, step=epoch)

        print(
            f"Epoch {epoch+1}/{epochs} "
            f"Loss {total_loss:.4f} "
            f"Train Acc {accuracy:.4f}"
        )

    return model, test_loader