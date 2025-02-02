import os
import pickle
from tqdm import tqdm
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .station_network import StationNetworkSimul

def create_degraded_networks(net_simul : StationNetworkSimul, df_flow, num_delete, num_degraded, data_dir):
    degraded_graphs = []
    for i in tqdm(range(num_degraded), total=num_degraded):
        new_net, removed_edges = net_simul.get_degraded_network(num_delete=num_delete)
        hash_removed  = hash(tuple(sorted(removed_edges)))
        
        os.makedirs(os.path.join(data_dir, f'delete_{num_delete}'), exist_ok=True)
        degraded_graph_path  = os.path.join(data_dir, f'delete_{num_delete}', f'{hash_removed}.gpickle')
        if os.path.exists(degraded_graph_path):
            # print(f"Getting graph {hash_removed} from precomputed data")
            with open(degraded_graph_path, 'rb') as f:
                new_net = pickle.load(f)
        else:
            # print(f"Creating and saving graph {hash_removed}")
            net_simul.update_degraded_network_nodes_traffic(new_net, removed_edges, df_flow)
            with open(degraded_graph_path, 'wb') as f:
                pickle.dump(new_net, f)
        degraded_graphs.append(new_net)
    return degraded_graphs


def nx_to_pyg_data(network : nx.DiGraph, target_name, node_feature_names, edge_feature_names):
    node_feature_tensor = torch.tensor(
        [
            [node[feature_name] for feature_name in node_feature_names] 
            for _, node in sorted(network.nodes.data())], dtype=torch.float) \
        if node_feature_names is not None else torch.tensor([i for i in sorted(network.nodes)], dtype=torch.int)
    
    edge_idx_tensor = torch.tensor([
            [edge[0] for edge in network.edges],
            [edge[1] for edge in network.edges]
            ], dtype=torch.long)
    
    edge_feature_tensor = torch.tensor([
            [edge[2][feature_name] for feature_name in edge_feature_names] for edge in sorted(network.edges.data())
            ], dtype=torch.float) if edge_feature_names is not None else None
    
    target_tensor = torch.tensor([node[target_name] for _, node in sorted(network.nodes.data())], dtype=torch.float)
    data = Data(x=node_feature_tensor, edge_index=edge_idx_tensor, y=target_tensor, edge_attr=edge_feature_tensor)
    return data


def get_degraded_network_loader(degraded_networks, target_name, node_feature_names=None, edge_feature_names=None, **kwargs):
    degraded_networks_pyg = [
        nx_to_pyg_data(
            degraded_network,
            target_name=target_name,
            node_feature_names=node_feature_names,
            edge_feature_names=edge_feature_names,
            ) for degraded_network in degraded_networks
        ]
    return DataLoader(degraded_networks_pyg, **kwargs)