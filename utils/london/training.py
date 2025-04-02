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
        return torch.mean(torch.abs(pred - target) / (target + 1e-8))
    

def get_flow_forecasting_metrics(pred, target):
        mask_high = target > 200
        mask_low =  target <= 200
        high_target = target[mask_high]
        low_target = target[mask_low]
        high_pred = pred[mask_high]
        low_pred = pred[mask_low]

        MAPE_high_target = torch.mean(torch.abs(high_pred - high_target) / (high_target))
        MAE_low_target = torch.mean(torch.abs(low_pred - low_target))

        return float(MAPE_high_target), float(MAE_low_target)



def test_model(model, test_loader):
    loss_mae = 0
    loss_mape = 0
    MAPE_high_target = 0
    MAE_low_target = 0
    MAE = CustomMAELoss()
    MAPE = CustomMAPELoss()

    for graph, temporal_features in test_loader:
        temporal_features = temporal_features.squeeze()
        output = model(graph, temporal_features)
        target = graph.y
        loss_mae += MAE(output, target)
        loss_mape += MAPE(output, target)
        MAPE_high_target_ , MAE_low_target_ = get_flow_forecasting_metrics(output, target)
        MAPE_high_target += float(MAPE_high_target_)
        MAE_low_target += float(MAE_low_target_)

    print(f"Test MAE: {loss_mae / len(test_loader)}")
    print(f"MAPE for high targets: {MAPE_high_target / len(test_loader)}")
    print(f"MAE for low targets: {MAE_low_target/ len(test_loader)}")