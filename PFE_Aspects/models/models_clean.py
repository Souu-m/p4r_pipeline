import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing


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
                use_edge_features=True, use_aspect_embeddings=False, encoder=False):
        super(SENGR_GCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.use_transformer = use_transformer
        self.use_edge_features = use_edge_features
        self.use_aspect_embeddings = use_aspect_embeddings
        self.encoder = encoder
        self.device = device

        
        print(f"Initialized model: {num_users} users, {num_items} items")
        print(f"Embedding dim: {embedding_dim}, Layers: {num_layers}")
        
        # Trainable embeddings for users and items (collaborative filtering)
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        
        # Load semantic embeddings if provided
######### if you want to test without semantic embeddings just set semantic_embeddings_path to none   ############
        semantic_embeddings_path=r'PFE_Aspects\models\projected_item_embeddings_64.pt'
        if semantic_embeddings_path:
            print("Loading semantic embeddings from:", semantic_embeddings_path)
            self._load_semantic_embeddings(semantic_embeddings_path)
        
        # Stack of GCN layers
        self.gcn_layers = nn.ModuleList([
            GCNLayer(embedding_dim, embedding_dim) for _ in range(num_layers)
        ])
        

        # Fully connected layer for rating prediction
        self.out = nn.Linear(embedding_dim * 2, 1)

    def _load_semantic_embeddings(self, semantic_path):
        """Load semantic embeddings and replace item_embeddings.weight"""
        semantic_dict = torch.load(semantic_path, map_location='cpu')
        
        # Create embedding matrix - assuming item IDs are 0, 1, 2, ..., num_items-1
        embedding_matrix = torch.zeros(self.num_items, self.item_embeddings.embedding_dim)
        
        for item_id, embedding in semantic_dict.items():
            if item_id < self.num_items:  # Safety check
                embedding_matrix[item_id] = embedding
            else :
                print(f"Warning: Item ID {item_id} exceeds num_items {self.num_items}")
        
        # Replace the weight data directly
        self.item_embeddings.weight.data = embedding_matrix
        print(f"Loaded semantic embeddings for {len(semantic_dict)} items")

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

    def predict_base(self, user_indices, item_indices, train_data):
        user_indices = user_indices.to(self.device)
        item_indices = item_indices.to(self.device)
        
        updated_embeddings = self.forward(train_data.edge_index, train_data.edge_weight)
        
        # Separate user & item embeddings
        user_emb = updated_embeddings[user_indices]
        item_emb = updated_embeddings[item_indices]

        # Concatenate user and item embeddings and predict ratings
        combined = torch.cat([user_emb, item_emb], dim=1)
        return torch.sigmoid(self.out(combined)) * 4 + 1  # Scale output to [1, 5]