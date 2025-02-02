import torch
import torch.nn as nn
from torch_geometric.nn.conv import GCNConv, GATv2Conv
from torch_geometric.nn import Sequential


class StationFlowGCN(nn.Module):
    def __init__(self, num_nodes : int, input_dim: int, output_dim: int, embeddings=None, freeze=True):
        super(StationFlowGCN, self).__init__()

        self.use_embeddings = False if embeddings is None else True
        
        if self.use_embeddings:
            if isinstance(embeddings, torch.Tensor):
                self.node_emb = nn.Embedding.from_pretrained(embeddings, freeze=freeze)
            elif embeddings == 'initialized':
                self.node_emb = nn.Embedding(num_nodes, input_dim)
            else:
                raise(Exception)
        
        self.gcn = Sequential('x, edge_index, edge_weight',
            [(GCNConv(input_dim, 128), 'x, edge_index, edge_weight -> x'),
            nn.ReLU(),
            (GCNConv(128, 64), 'x, edge_index, edge_weight -> x'),
            nn.ReLU(),
            (GCNConv(64, 64), 'x, edge_index, edge_weight -> x'),
            nn.ReLU(),
            (GCNConv(64, 32), 'x, edge_index, edge_weight -> x'),
            nn.ReLU(),
            (GCNConv(32, 32), 'x, edge_index, edge_weight -> x'),
            nn.ReLU(),
            (GCNConv(32, output_dim), 'x, edge_index, edge_weight -> x'),
            nn.ReLU()],
        )

    def forward(self,x, edge_index, edge_weight=None):
        
        if self.use_embeddings:
            x = self.node_emb(x)

        x = self.gcn(x, edge_index, edge_weight)
        return x


class StationFlowGAT(nn.Module):
    def __init__(self, num_nodes : int, input_dim: int, output_dim: int, edge_dim=None, embeddings=None, freeze=True, num_heads=1):
        super(StationFlowGAT, self).__init__()

        self.use_embeddings = False if embeddings is None else True
        
        if self.use_embeddings:
            if isinstance(embeddings, torch.Tensor):
                self.node_emb = nn.Embedding.from_pretrained(embeddings, freeze=freeze)
            elif embeddings == 'initialized':
                self.node_emb = nn.Embedding(num_nodes, input_dim)
            else:
                raise(Exception)
            
        self.gat = Sequential('x, edge_index, edge_attr',
            [(GATv2Conv(input_dim, 64, heads=num_heads, edge_dim=edge_dim), 'x, edge_index, edge_attr -> x'),
            nn.ReLU(),
            (GATv2Conv(64*num_heads, 32, heads=num_heads, edge_dim=edge_dim), 'x, edge_index, edge_attr -> x'),
            nn.ReLU(),
            (GATv2Conv(32*num_heads, 32, heads=num_heads, edge_dim=edge_dim), 'x, edge_index, edge_attr -> x'),
            nn.ReLU(),
            (GATv2Conv(32*num_heads, output_dim, edge_dim=edge_dim), 'x, edge_index, edge_attr -> x'),
            nn.ReLU()],
        )

    def forward(self,x, edge_index, edge_attr=None):
        
        if self.use_embeddings:
            x = self.node_emb(x)
        x = self.gat(x, edge_index, edge_attr)     
        
        return x  



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