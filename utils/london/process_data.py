
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader


class GraphSequenceDataset(Dataset):
    def __init__(self, graphs, window_size=4):
        self.graphs = graphs
        self.window_size = window_size
        if len(graphs) <= window_size:
            raise ValueError("Error : window_size should be lower than the number of graphs")
        
    def __len__(self):
        return len(self.graphs) - self.window_size

    def __getitem__(self, idx):
        # get last flows
        last_graphs = self.graphs[idx : idx + self.window_size]
        last_flows = [graph.y for graph in last_graphs]
        last_flows = torch.cat(last_flows, dim=1)

        # add last flows as nodes features
        idx_graph = self.graphs[idx + self.window_size].clone()
        temporal_features = idx_graph.x
        idx_graph.x = last_flows
        return idx_graph, temporal_features
    

def build_train_test_loaders(graphs, window_size, train_split=0.8):
    train_size = int(train_split * len(graphs))

    train_dataset = GraphSequenceDataset(graphs[:train_size], window_size)  
    test_dataset = GraphSequenceDataset(graphs[train_size:], window_size)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, test_loader


# For second version with flows only between 0000-0500

class GraphSequenceDayDataset(Dataset):
    def __init__(self, graphs, window_size=4):
        self.graphs = graphs
        self.window_size = window_size
        self.accesible_data_per_day = len(graphs[0]) - self.window_size
        if self.accesible_data_per_day < 0:
            print(len(graphs[0]))
            raise ValueError("Error : window_size should be lower than the number of graphs per day")
        
    def __len__(self):
        return len(self.graphs) * self.accesible_data_per_day

    def __getitem__(self, idx):
        day = idx//self.accesible_data_per_day - 1
        qh = idx%self.accesible_data_per_day

        day_graphs = self.graphs[day]

        # get last flows
        last_graphs = day_graphs[qh : qh + self.window_size]
        last_flows = [graph.y for graph in last_graphs]
        last_flows = torch.cat(last_flows, dim=1)

        # add last flows as nodes features
        idx_graph = day_graphs[qh + self.window_size].clone()
        temporal_features = idx_graph.x
        idx_graph.x = last_flows
        return idx_graph, temporal_features
    

def build_train_test_loaders_2(graphs, window_size, train_split=0.8, batch_size=1, shuffle=True):
    train_size = int(train_split * len(graphs))

    train_dataset = GraphSequenceDayDataset(graphs[:train_size], window_size)  
    test_dataset = GraphSequenceDayDataset(graphs[train_size:], window_size)

    train_loader = DataLoader(train_dataset, batch_size, shuffle)
    test_loader = DataLoader(test_dataset, batch_size, shuffle)

    return train_loader, test_loader