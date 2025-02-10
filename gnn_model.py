import torch
import torch.nn as nn
from torch_geometric.nn.conv import GCNConv, GATv2Conv
from torch_geometric.nn import Sequential
from torch_geometric.nn import APPNP

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
        
        self.conv = Sequential('x, edge_index, edge_weight',
            [(GCNConv(input_dim, 128), 'x, edge_index, edge_weight -> x'),
            nn.ReLU(),
            (GCNConv(128, 64), 'x, edge_index, edge_weight -> x'),
            nn.ReLU(),
            (GCNConv(64, 64), 'x, edge_index, edge_weight -> x'),
            nn.ReLU(),
            (GCNConv(64, 32), 'x, edge_index, edge_weight -> x'),
            nn.ReLU(),
            (GCNConv(32, 32), 'x, edge_index, edge_weight -> x'),
            nn.ReLU()]
        )

        self.node_head = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.ReLU(),
        )

        self.edge_head = nn.Sequential(
            nn.Linear(2 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.ReLU(),
        )

    def forward(self,x, edge_index, edge_weight=None):
        
        if self.use_embeddings:
            x = self.node_emb(x)

        x = self.conv(x, edge_index, edge_weight)
        x_e = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)

        node_output = self.node_head(x)
        edge_output = self.edge_head(x_e)

        return node_output, edge_output


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
            nn.ReLU(),],
        )
        
        self.node_head = nn.Sequential(
            nn.Linear(32*num_heads, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.ReLU(),
        )

        self.edge_head = nn.Sequential(
            nn.Linear(2 * 32*num_heads, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.ReLU(),
        )

    def forward(self,x, edge_index, edge_attr=None):
        
        if self.use_embeddings:
            x = self.node_emb(x)

        x = self.gat(x, edge_index, edge_attr)     
        x_e = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)

        node_output = self.node_head(x)
        edge_output = self.edge_head(x_e)
        
        return node_output, edge_output


# Définition du modèle GCN avec régression quantile
class QuantileGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, quantiles=[0.05, 0.5, 0.95]):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, len(quantiles))  # 1 sortie par quantile
        self.quantiles = quantiles

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x  # Retourne [nœuds, quantiles]
    



class StationFlowGCN2(nn.Module):
    def __init__(self, num_nodes : int, input_dim: int, output_dim: int, embeddings=None, freeze=True):
        super(StationFlowGCN2, self).__init__()

        self.use_embeddings = False if embeddings is None else True
        
        if self.use_embeddings:
            if isinstance(embeddings, torch.Tensor):
                self.node_emb = nn.Embedding.from_pretrained(embeddings, freeze=freeze)
            elif embeddings == 'initialized':
                self.node_emb = nn.Embedding(num_nodes, input_dim)
            else:
                raise(Exception)
        
        self.gcn = Sequential('x, edge_index, edge_weight',
            [(GCNConv(input_dim, 256), 'x, edge_index, edge_weight -> x'),
            nn.ReLU(),
            (GCNConv(256, 128), 'x, edge_index, edge_weight -> x'),
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
    



class StationFlowGCN3(nn.Module):
    def __init__(self, num_nodes : int, input_dim: int, output_dim: int, embeddings=None, freeze=True):
        super(StationFlowGCN3, self).__init__()

        self.use_embeddings = False if embeddings is None else True
        
        if self.use_embeddings:
            if isinstance(embeddings, torch.Tensor):
                self.node_emb = nn.Embedding.from_pretrained(embeddings, freeze=freeze)
            elif embeddings == 'initialized':
                self.node_emb = nn.Embedding(num_nodes, input_dim)
            else:
                raise(Exception)
        
        self.gcn = Sequential('x, edge_index, edge_weight',
            [(GCNConv(input_dim, 16), 'x, edge_index, edge_weight -> x'),
            nn.ReLU(),
            (GCNConv(16, 8), 'x, edge_index, edge_weight -> x'),
            nn.ReLU(),
            (GCNConv(8, 8), 'x, edge_index, edge_weight -> x'),
            nn.ReLU(),
            (GCNConv(8, output_dim), 'x, edge_index, edge_weight -> x'),
            nn.ReLU()],
        )

    def forward(self,x, edge_index, edge_weight=None):
        
        if self.use_embeddings:
            x = self.node_emb(x)

        x = self.gcn(x, edge_index, edge_weight)
        return x
    

class DeepGCN(torch.nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, output_dim, embeddings=None, freeze=True):
        super(DeepGCN, self).__init__()
        # self.convs = torch.nn.ModuleList()

        self.use_embeddings = False if embeddings is None else True
        
        if self.use_embeddings:
            if isinstance(embeddings, torch.Tensor):
                self.node_emb = nn.Embedding.from_pretrained(embeddings, freeze=freeze)
            elif embeddings == 'initialized':
                self.node_emb = nn.Embedding(num_nodes, input_dim)
            else:
                raise(Exception)
        

        self.gcn = Sequential('x, edge_index, edge_weight',
            [(GCNConv(input_dim, 16), 'x, edge_index, edge_weight -> x'),
            nn.ReLU(),
            (GCNConv(16, 16), 'x, edge_index, edge_weight -> x'),
            nn.ReLU(),
            (GCNConv(16, 16), 'x, edge_index, edge_weight -> x'),
            nn.ReLU(),
            (GCNConv(16, 16), 'x, edge_index, edge_weight -> x'),
            nn.ReLU(),
            (GCNConv(16, 16), 'x, edge_index, edge_weight -> x'),
            nn.ReLU(),
            (GCNConv(16, 16), 'x, edge_index, edge_weight -> x'),
            nn.ReLU(),
            (GCNConv(16, 16), 'x, edge_index, edge_weight -> x'),
            nn.ReLU(),
            (GCNConv(16, 16), 'x, edge_index, edge_weight -> x'),
            nn.ReLU(),
            (GCNConv(16, 16), 'x, edge_index, edge_weight -> x'),
            nn.ReLU(),
            (GCNConv(16, 16), 'x, edge_index, edge_weight -> x'),
            nn.ReLU(),
            (GCNConv(16, 16), 'x, edge_index, edge_weight -> x'),
            nn.ReLU(),
            (GCNConv(16, 16), 'x, edge_index, edge_weight -> x'),
            nn.ReLU(),
            (GCNConv(16, output_dim), 'x, edge_index, edge_weight -> x'),
            nn.ReLU()],
        )


    def forward(self, x, edge_index, edge_weight=None):
        if self.use_embeddings:
            x = self.node_emb(x)

        x = self.gcn(x, edge_index, edge_weight)
        return x
    

class GraphAPPNP(torch.nn.Module):
    def __init__(self, K, alpha, input_dim, output_dim):
        super(GraphAPPNP, self).__init__()
        self.K = K
        self.alpha = alpha
        self.appnp = Sequential('x, edge_index', [
            (GCNConv(input_dim, 128), 'x, edge_index -> x'),
            nn.ReLU(),
            (APPNP(K=K, alpha=alpha), 'x, edge_index -> x'),  # Propagation longue distance
            (GCNConv(128, output_dim), 'x, edge_index -> x')
        ])

    def forward(self, x, edge_index, edge_weight=None):
        return self.appnp(x, edge_index)
