import torch
import torch.nn as nn
from torch_geometric.nn.conv import GCNConv


class StationFlowGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(StationFlowGNN, self).__init__()
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


class InterStationFlowGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(InterStationFlowGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),  # Combinaison des deux nœuds reliés par une arête
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)      # Prédiction finale pour chaque arête
        )
        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # Étape 1 : Propagation des caractéristiques des nœuds
        x = self.relu(self.conv1(x, edge_index))  # Première couche GCN
        x = self.relu(self.conv2(x, edge_index))  # Deuxième couche GCN

        # Étape 2 : Construction des représentations pour les arêtes
        # edge_index[0] = indices des nœuds sources
        # edge_index[1] = indices des nœuds cibles
        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)  # [E, 2*hidden_dim]

        # Étape 3 : Prédictions pour les arêtes
        edge_predictions = self.edge_mlp(edge_features)  # [E, output_dim]

        return edge_predictions