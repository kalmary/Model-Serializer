import torch
from sklearn.metrics import confusion_matrix, roc_curve


def evaluate(model, dataloader):

    model.eval()

    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():

        for X, y in dataloader:

            probs = model(X)
            predicted = (probs > 0.5).float()

            correct += (predicted == y).sum().item()
            total += y.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = correct / total
    cm = confusion_matrix(all_labels, all_preds)
    rc = roc_curve(all_labels, all_probs)

    return accuracy, cm, rc