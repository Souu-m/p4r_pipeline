import copy
import json
import os
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from config.config import Config
from models.models_clean import SENGR_GCN
from torch_geometric.loader import DataLoader


from util.evaluation_utils import evaluate_model, print_metrics
from util.graph_preparation_utils import prepare_data, split_and_prepare_data
from util.train_utils import train_model

import copy
from datetime import datetime
import json

def set_seed(seed=14):
    # Python's built-in random module
    random.seed(seed)
    
    # Numpy's random module
    np.random.seed(seed)
    
    # PyTorch seed for CPU
    torch.manual_seed(seed)
    
    # PyTorch seed for all GPU devices (if using CUDA)
    torch.cuda.manual_seed_all(seed)
    
    # Make sure to disable CuDNN's non-deterministic optimizations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def train_and_evaluate(config, train_data, val_data, test_data, n_users, n_items, 
                                embedding_dim=None, num_layers=None,transformer_layers=None,transformer_heads=None ):
    """Train global model (original graph)  and evaluate on test set"""
    
    
    # Use provided params or defaults from config
    if embedding_dim is None:
        embedding_dim = config.EMBEDDING_DIM
    if num_layers is None:
        num_layers = config.NUM_LAYERS
    if transformer_layers is None:
        transformer_layers = config.TRANSFORMER_LAYERS
    if transformer_heads is None:
        transformer_heads = config.TRANSFORMER_HEADS
    print(f"Using embedding_dim={embedding_dim}, num_layers={num_layers}")

    
    
    print("\n=== Initializing model ===")
    model = SENGR_GCN(
        n_users, n_items, 
        embedding_dim=embedding_dim, 
        num_layers=num_layers,
        use_transformer=config.USE_TRANSFORMER,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        use_edge_features=config.USE_EDGE_FEATURES,
        device=config.DEVICE,
    ).to(config.DEVICE) 

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.MSELoss()
    
    g = torch.Generator()
    g.manual_seed(config.SEED)
    loader = DataLoader([train_data], batch_size=config.BATCH_SIZE, shuffle=True, generator=g)
    
    print("\n=== Training model ===")
    model = train_model(
        model, optimizer, loader, val_data, criterion, 
        config,
        n_users=n_users,
        num_epochs=config.NUM_EPOCHS, 
        patience=config.PATIENCE
    )
    
    print("\n=== Evaluating on test set ===")
    test_metrics = evaluate_model(model, test_data,train_data=train_data ,k=config.k, threshold=config.Threshold, n_users=n_users)
    
    return model, test_metrics



def main_global(config): # run original graph which mean global mode (without aspects)
    """Run single experiment with config settings"""
    print("embedding size",config.EMBEDDING_DIM)
    print("number of layers",config.NUM_LAYERS)
    # Prepare data
    ui_edges, ui_weights, social_edges_list, n_users, n_items, builder = prepare_data(config)
    train_data, val_data, test_data = split_and_prepare_data(ui_edges,ui_weights,social_edges_list ,builder, config,n_users, n_items)
    
    # Train and evaluate
    model, test_metrics = train_and_evaluate(
        config, train_data, val_data, test_data, n_users, n_items
    )
    
    print_metrics(test_metrics, k=config.k)
    
    return model, test_metrics




if __name__ == "__main__":
    config = Config()
    set_seed(config.SEED)  # Set random seed for reproducibility
    print("Using device:", config.DEVICE)

    # start  with embedding_dim=64, num_layers=2  then when results imporves inchlh try with embedding_dim=64, num_layers=2
    model, metrics = main_global(config) # train and evaluate the global model (Base model with original graph that we aim to improve with aspects)

