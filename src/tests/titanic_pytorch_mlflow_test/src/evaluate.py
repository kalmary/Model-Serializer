import torch


def evaluate(model, dataloader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for X, y in dataloader:

            preds = model(X)

            predicted = (preds > 0.5).float()

            correct += (predicted == y).sum().item()
            total += y.size(0)

    accuracy = correct / total

    return accuracy