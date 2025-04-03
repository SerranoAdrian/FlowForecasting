import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class STGCN(nn.Module):
    def __init__(self, node_features, hidden_dim, temporal_features_dim=10):
        super(STGCN, self).__init__()
        self.gcn = GCNConv(node_features, hidden_dim)  # GCN
        self.fc1 = nn.Linear(hidden_dim + temporal_features_dim, hidden_dim)  # Prédiction finale
        self.fc2 = nn.Linear(hidden_dim + node_features, 1)
        # self.bn1 = nn.BatchNorm1d(hidden_dim)

    def forward(self, graph, temporal_features):
        x = self.gcn(graph.x, graph.edge_index)  # GCN
        x = F.relu(x) # (num_nodes, hideen_dim)
        
        # Use temporal features and x as inputs of fc and use the fc layer
        x = torch.cat((x, temporal_features), dim=-1)
        x = self.fc1(x) 
        x = F.relu(x)
        x = torch.cat((x, graph.x), dim=-1)
        x = self.fc2(x) 
        x = F.relu(x)
        return x # return the flow for each nodes (num_nodes, 1)
    


class ImprovedSTGCN(nn.Module):
    def __init__(self, node_features, hidden_dim, temporal_features_dim=10):
        super(ImprovedSTGCN, self).__init__()
        self.gcn = GCNConv(node_features, hidden_dim)  # GCN
        self.fc1 = nn.Linear(hidden_dim + temporal_features_dim, hidden_dim)  # Prédiction finale
        self.fc2 = nn.Linear(hidden_dim + node_features, 1)
        # self.bn1 = nn.BatchNorm1d(hidden_dim)

    def forward(self, graph, temporal_features):
        x = self.gcn(graph.x, graph.edge_index)  # GCN
        x = F.relu(x) # (num_nodes, hideen_dim)
        
        # Use temporal features and x as inputs of fc and use the fc layer
        x = torch.cat((x, temporal_features), dim=-1)
        x = self.fc1(x) 
        x = F.relu(x)
        x = torch.cat((x, graph.x), dim=-1)
        x = self.fc2(x) 
        x = F.relu(x)
        return x # return the flow for each nodes (num_nodes, 1)