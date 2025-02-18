import torch
import torch.nn as nn


class CustomMAELoss(nn.Module):
    def __init__(self):
        super(CustomMAELoss, self).__init__()

    def forward(self, pred, target):
        return torch.mean(torch.abs(pred - target))


class CustomMAPELoss(nn.Module):
    def __init__(self):
        super(CustomMAPELoss, self).__init__()

    def forward(self, pred, target):
        return torch.mean(torch.abs(pred - target) / (target + 1))
    

def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    with torch.no_grad():  # Désactivation des gradients pour l'évaluation
        for graph, temporal_features in test_loader:
            temporal_features = temporal_features.squeeze()  # Squeeze si nécessaire
            output = model(graph, temporal_features)  # Assure-toi de passer aussi les temporal_features
            target = graph.y
            loss = criterion(output, target)
            test_loss += loss.item()

    print(f"Test loss: {test_loss / len(test_loader)}")