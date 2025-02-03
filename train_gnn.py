import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch.optim as optim
from collections import defaultdict

### TRAIN & EVAL

def train_node_gnn_model(node_gnn_model, config, train_loader, dev_loader):
    
    epochs = config['epochs']
    lr = config['lr']
    criterion = config['criterion']

    optimizer = optim.Adam(node_gnn_model.parameters(), lr=lr)


    for epoch in range(epochs):
        node_gnn_model.train()
        train_loss = []
        node_gnn_model.train()
        for data in train_loader:
            optimizer.zero_grad()
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
            output = node_gnn_model(x, edge_index, edge_weight)
            loss = criterion(output.squeeze(), data.y)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        
        eval_metrics = eval_node_gnn_model(node_gnn_model, dev_loader, config)
        metrics_result = "\t".join([f"{metric_name}: {metric_value}" for metric_name, metric_value in eval_metrics.items()])
        print()
        print(f'Epoch {epoch} -', f'Train loss: {np.mean(train_loss)}', 'Eval:', metrics_result)
        

def eval_node_gnn_model(node_gnn_model, loader, config):
    eval_metrics = {metric_name : [] for metric_name in config['metrics'].keys()}
    node_gnn_model.eval()
    with torch.no_grad():
        for data in loader:
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
            output = node_gnn_model(x, edge_index, edge_weight)
            for metric_name, metric in config['metrics'].items():
                eval_metrics[metric_name].append(metric(output.squeeze(), data.y))
        
        for metric_name in config['metrics'].keys():
            eval_metrics[metric_name] = np.mean(eval_metrics[metric_name])
        
        return eval_metrics

# def train_edge_gnn_model(edge_gnn_model, config, train_loader, dev_loader):
    
#     epochs = config['epochs']
#     lr = config['lr']
#     criterion = config['criterion']

#     optimizer = optim.Adam(edge_gnn_model.parameters(), lr=lr)


#     for epoch in range(epochs):
#         edge_gnn_model.train()
#         train_loss = []
#         edge_gnn_model.train()
#         for data in train_loader:
#             optimizer.zero_grad()
#             x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
#             output = node_gnn_model(x, edge_index, edge_weight)
#             loss = criterion(output.squeeze(), data.y)
#             train_loss.append(loss.item())
#             loss.backward()
#             optimizer.step()
        
#         eval_metrics = eval_node_gnn_model(node_gnn_model, dev_loader, config)
#         metrics_result = "\t".join([f"{metric_name}: {metric_value}" for metric_name, metric_value in eval_metrics.items()])
#         print()
#         print(f'Epoch {epoch} -', f'Train loss: {np.mean(train_loss)}', 'Eval:', metrics_result)
        

# def eval_edge_gnn_model(node_gnn_model, loader, config):
#     eval_metrics = {metric_name : [] for metric_name in config['metrics'].keys()}
#     node_gnn_model.eval()
#     with torch.no_grad():
#         for data in loader:
#             x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
#             output = node_gnn_model(x, edge_index, edge_weight)
#             for metric_name, metric in config['metrics'].items():
#                 eval_metrics[metric_name].append(metric(output.squeeze(), data.y))
        
#         for metric_name in config['metrics'].keys():
#             eval_metrics[metric_name] = np.mean(eval_metrics[metric_name])
        
#         return eval_metrics
### METRICS

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

def get_metric_per_node(node_model, data, metric):
    x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
    output = node_model(x, edge_index, edge_weight)
    metric_nodes = metric(output.squeeze(), data.y).detach().numpy()
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

### PLOT

def boxplot_node_metric(df, node_idx, network_simul, metric_name):
    plt.figure(figsize=(3,7))
    sns.boxplot(df.iloc[node_idx][:-1], showmeans=True)
    plt.xlabel(f"{network_simul.network_graph.nodes[node_idx]['title']} (ligne {network_simul.network_graph.nodes[node_idx]['group']})")
    plt.ylabel(f'{metric_name}')

def boxplot_node_metric_per_line(df, line, network_simul, metric_name):
    plt.figure(figsize=(10,5))
    df_boxplot_line = df[df['line'] == str(line)].drop(columns='line')
    df_boxplot_line = df_boxplot_line.T
    boxplot = sns.boxplot(df_boxplot_line, orient='h', showmeans=True)
    boxplot.set_yticklabels((network_simul.network_graph.nodes[node_idx]['title'] for node_idx in df_boxplot_line.columns))
    plt.xlabel(f'{metric_name}')

def plot_true_predicted(predicted_flows, actual_flows):
    plt.figure(figsize=(5,2.5))
    sns.scatterplot(x=actual_flows, y=predicted_flows)
    plt.plot([min(actual_flows), max(actual_flows)],
            [min(actual_flows), max(actual_flows)],
            color='red', linestyle='--')
    plt.xlabel("Actual Passenger Flow")
    plt.ylabel("Predicted Passenger Flow")
    plt.title("Actual vs Predicted Passenger Flow")

def plot_predicted_ape(predicted_flows, actual_flows):
    mape = MAPE_loss(reduction='none')
    ape_predictions = mape(predicted_flows, actual_flows)
    
    plt.figure(figsize=(5,2.5))
    sns.scatterplot(x=actual_flows, y=ape_predictions)
    plt.plot([min(actual_flows), max(actual_flows)],
            [0, 0],
            color='red', linestyle='--')  # Perfect match line
    plt.xlabel("Actual Passenger Flow")
    plt.ylabel("Absolute Percentage Error")
    plt.title("APE with respect to the reference passenger flow")