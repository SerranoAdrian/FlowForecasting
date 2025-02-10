import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

def train_gnn_model(node_gnn_model, config, train_loader, dev_loader):
    
    epochs = config['epochs']
    lr = config['lr']
    node_criterion = config['criterion'].get('node')
    edge_criterion = config['criterion'].get('edge')
    regul_criterion = config['criterion'].get('regul')

    optimizer = optim.Adam(node_gnn_model.parameters(), lr=lr)

    train_loss = {
        type_loss : [] for type_loss in config['criterion'].keys()
    }
    for epoch in range(epochs):
        node_gnn_model.train()
        epoch_loss = {
                type_loss : [] for type_loss in config['criterion'].keys()
            }
        node_gnn_model.train()
        for data in train_loader:
            optimizer.zero_grad()
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
            output = node_gnn_model(x, edge_index, edge_weight)
            loss = 0

            if node_criterion is not None:
                node_loss = node_criterion(output[0].squeeze(), data.y)
                loss += node_loss
                epoch_loss['node'].append(node_loss.detach())
            if edge_criterion is not None:
                edge_loss = edge_criterion(output[1].squeeze(), data.ye)
                loss += edge_loss
                epoch_loss['edge'].append(edge_loss.detach())
            if regul_criterion is not None:
                regul_loss = regul_criterion(output, data)
                loss += 0.1*regul_loss
                epoch_loss['regul'].append(regul_loss.detach())

            loss.backward()
            optimizer.step()
        
        for type_loss in train_loss.keys():
            train_loss[type_loss].append(np.mean(epoch_loss[type_loss]))
        train_results = " - ".join([f'{type_loss}: {np.mean(loss_values)}' for type_loss, loss_values in train_loss.items()])
        
        eval_metrics = eval_gnn_model(node_gnn_model, dev_loader, config)
        metric_results_node = "\t".join([f"{metric_name}: {metric_value}" for metric_name, metric_value in eval_metrics['node'].items()])
        metric_results_edge = "\t".join([f"{metric_name}: {metric_value}" for metric_name, metric_value in eval_metrics['edge'].items()])
        metric_results = f'Node - {metric_results_node}\nEdge - {metric_results_edge}'
        
        print('Epoch', epoch)
        print('TRAIN:', train_results)
        print('EVAL:', metric_results)
        print()

    for type_loss in train_loss.keys():
        plt.plot(train_loss[type_loss], label=type_loss)

    plt.xlabel('Epochs')
    plt.legend()
        

def eval_gnn_model(node_gnn_model, loader, config):
    eval_metrics = {
        type_loss : {metric_name : [] for metric_name in config['metrics'].keys()}
        for type_loss in ('node', 'edge')
        }
    node_gnn_model.eval()
    with torch.no_grad():
        for data in loader:
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
            node_output, edge_output = node_gnn_model(x, edge_index, edge_weight)
            for metric_name, metric in config['metrics'].items():
                
                eval_metrics['node'][metric_name].append(metric(node_output.squeeze(), data.y))
                eval_metrics['edge'][metric_name].append(metric(edge_output.squeeze(), data.ye))
        
        for metric_name in config['metrics'].keys():
            eval_metrics['node'][metric_name]= np.mean(eval_metrics['node'][metric_name])
            eval_metrics['edge'][metric_name]= np.mean(eval_metrics['edge'][metric_name])
        
        return eval_metrics
    
def regul_edge_node_flow(reduction='mean', norm='l1'):
    def regul(output, data):
        
        if norm=='l1':
            criterion = torch.nn.L1Loss()
        elif norm=='l2':
            criterion = torch.nn.MSELoss() 
        else:
            raise(Exception())

        # _, edge_output = output
        _, edge_output = output
        edge_index = data.edge_index

        incoming_edges_sum_pred = torch.zeros_like(data.y)
        target_nodes = edge_index[1]
        incoming_edges_sum_pred.index_add_(0, target_nodes, edge_output.squeeze())
        regul_node_edge = criterion(data.y, incoming_edges_sum_pred)
        return regul_node_edge
    
    return regul