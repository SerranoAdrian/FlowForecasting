import torch
import torch.nn as nn
from torch_geometric.nn.conv import GCNConv


class PassengerFlowGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PassengerFlowGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)  # Output_dim = 1 pour chaque nœud
        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # Propagation à travers les couches GNN
        x = self.relu(self.conv1(x, edge_index))  # Première couche GCN
        x = self.relu(self.conv2(x, edge_index))  # Deuxième couche GCN
        # Projection sur une seule dimension pour chaque nœud (flux de passagers)
        x = self.fc(x)
        return x  # Sortie de dimension [N, 1], où N est le nombre de nœuds
