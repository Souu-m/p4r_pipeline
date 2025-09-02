"""Enhanced GCN model with semantic item embeddings integration - Corrected TransformerConv Version"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import TransformerConv


# GCN implementation
class GCNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__(aggr='add')
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        aggregated = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        updated = (aggregated + x) / 2
        return self.linear(updated)

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


class SENGR_GCN(nn.Module):
    def __init__(self, num_users, num_items, device, embedding_dim=64, num_layers=2, 
                use_transformer=False, transformer_layers=1, transformer_heads=2,
                use_edge_features=True, encoder=False):
        super(SENGR_GCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.use_transformer = use_transformer
        self.use_edge_features = use_edge_features
        self.encoder = encoder
        self.device = device

        
        print(f"Initialized model: {num_users} users, {num_items} items")
        print(f"Embedding dim: {embedding_dim}, Layers: {num_layers}")
        
        # Trainable embeddings for users and items (collaborative filtering)
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        

        # Stack of GCN layers
        self.gcn_layers = nn.ModuleList([
            GCNLayer(embedding_dim, embedding_dim) for _ in range(num_layers)
        ])
        

        # Fully connected layer for rating prediction
        self.out = nn.Linear(embedding_dim * 2, 1)


    def forward(self, edge_index, edge_weight):
        """
        Forward pass through GCN layers with transformer enhancement.
        """
        edge_index = edge_index.to(self.device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(self.device)
        
        # Combine user and item embeddings
        user_item_embeddings = torch.cat([
            self.user_embeddings.weight,
            self.item_embeddings.weight
        ], dim=0)
        
        # Pass through GCN layers
        for layer in self.gcn_layers:
            user_item_embeddings = layer(user_item_embeddings, edge_index, edge_weight)
            

        return user_item_embeddings

    def predict_base(self, user_indices, item_indices, train_data, mode=None):
        user_indices = user_indices.to(self.device)
        item_indices = item_indices.to(self.device)
        
        updated_embeddings = self.forward(train_data.edge_index, train_data.edge_weight)
        
        # Separate user & item embeddings
        user_emb = updated_embeddings[user_indices]
        item_emb = updated_embeddings[item_indices]

        # Concatenate user and item embeddings and predict ratings
        combined = torch.cat([user_emb, item_emb], dim=1)
        return torch.sigmoid(self.out(combined)) * 4 + 1  # Scale output to [1, 5]