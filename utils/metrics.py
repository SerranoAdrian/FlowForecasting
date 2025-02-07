import torch
from collections import defaultdict
import pandas as pd

def MAPE_loss(reduction='mean'):
    def MAPE(y_pred, y_true):
        """
        Compute the Mean Absolute Percentage Error for the model predictions.

        Inputs:
            - y_pred : model predictions
            - y_true : true values
        
        Return:
            MAPE : Mean Absolute Percentage Error
        """
        APE = torch.abs(y_pred - y_true)/torch.abs(y_true)
        if reduction == 'mean':
            APE = torch.mean(APE)
        return APE
    return MAPE

def WAPE_loss(reduction='mean'):
    def WAPE(y_pred, y_true):
        """
        Compute the Mean Absolute Percentage Error for the model predictions.

        Inputs:
            - y_pred : model predictions
            - y_true : true values
        
        Return:
            WAPE : Mean Absolute Percentage Error
        """
        APE = torch.abs(y_pred - y_true)/torch.sum(y_true)
        if reduction == 'mean':
            APE = torch.mean(APE)
        return APE
    return WAPE

def WMAPE_loss(reduction='mean'):
    def WMAPE(y_pred, y_true):
        """
        Compute the Mean Absolute Percentage Error for the model predictions.

        Inputs:
            - y_pred : model predictions
            - y_true : true values
        
        Return:
            WMAPE : Mean Absolute Percentage Error
        """
        weights = y_true/y_true.max()
        APE = (weights*torch.abs(y_pred - y_true))/torch.sum(weights*y_true)
        if reduction == 'mean':
            APE = torch.mean(APE)
        return APE
    return WMAPE

def get_metric_per_node(node_model, data, metric):
    x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
    node_output, _ = node_model(x, edge_index, edge_weight)
    metric_nodes = metric(node_output.squeeze(), data.y).detach().numpy()
    return metric_nodes


def get_metric_per_node_per_network(nodes_gnn_model, loader, metric, network_simul):
    metric_per_node_per_network = defaultdict(lambda : [])
    for data in loader:
        metric_per_node = get_metric_per_node(nodes_gnn_model, data, metric)
        for node_idx, metric_node in enumerate(metric_per_node):
            metric_per_node_per_network[node_idx].append(metric_node)
    num_nodes = len(network_simul.network_graph)
    for node in range(num_nodes):
        metric_per_node_per_network[node].append(network_simul.network_graph.nodes[node]['group'])

    df = pd.DataFrame(metric_per_node_per_network)
    df = df.T
    df = df.rename(columns={len(loader) : 'line'})
    return df