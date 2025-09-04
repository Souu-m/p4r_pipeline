"""Data loading and splitting utilities"""


import numpy as np
from sklearn.preprocessing import StandardScaler  # Fixed import
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import os
import ast
import re
from typing import List, Any
from collections import defaultdict

def load_data(config):
    """Load and preprocess data based on configuration"""
    from preprocessing import Preprocessor
    
    # Load data
    business_df = pd.read_csv(config.BUSINESS_DATA_PATH)
    user_df = pd.read_csv(config.USER_DATA_PATH)
    review_df = pd.read_csv(config.REVIEW_DATA_PATH)
    
    return user_df, business_df, review_df


def split_data(edge_index, edge_weight, train_rate=0.8, random_state=1):
    """
    Split edges into train / val / test for a regression task.
    
    Returns:
        (train_edge_index, train_edge_weight),
        (val_edge_index, val_edge_weight),
        (test_edge_index, test_edge_weight)
    """
    n_edges = edge_index.size(1)
    all_indices = list(range(n_edges))
    
    # Split into train and temp (val + test)
    train_indices, temp_indices = train_test_split(
        all_indices, 
        train_size=train_rate, 
        random_state=random_state
    )
    
    # Split temp into val and test (50-50)
    val_indices, test_indices = train_test_split(
        temp_indices, 
        test_size=0.5, 
        random_state=random_state
    )
    
    # Create splits
    train_split = (edge_index[:, train_indices], edge_weight[train_indices])
    val_split = (edge_index[:, val_indices], edge_weight[val_indices])
    test_split = (edge_index[:, test_indices], edge_weight[test_indices])
    
    return train_split, val_split, test_split


def extract_user_features(user_df, user_encoder):
    user_df = user_df.copy()
    user_df = user_df[user_df["user_id"].isin(user_encoder.classes_)]
    
    # Check if we have any users after filtering
    if len(user_df) == 0:
        print("Warning: No users found after filtering with encoder classes.")
        # Return zero tensor with appropriate shape
        return torch.zeros((len(user_encoder.classes_), 3))
    
    user_df["encoded_id"] = user_encoder.transform(user_df["user_id"])

    # Use average_stars, review_count, fans
    features = user_df[["encoded_id", "average_stars", "review_count", "fans"]].fillna(0)
    
    # Check if we have data to scale
    if len(features) == 0:
        print("Warning: No features to scale after preprocessing.")
        return torch.zeros((len(user_encoder.classes_), 3))
    
    # Only scale if we have data
    scaler = StandardScaler()
    feature_cols = ["average_stars", "review_count", "fans"]
    features[feature_cols] = scaler.fit_transform(features[feature_cols])
    
    # Create feature tensor
    feature_tensor = torch.zeros((len(user_encoder.classes_), len(feature_cols)))
    for _, row in features.iterrows():
        encoded_id = int(row["encoded_id"])
        if encoded_id < len(user_encoder.classes_):  # Safety check
            feature_tensor[encoded_id] = torch.tensor(row[1:].values, dtype=torch.float)
    
    return feature_tensor


def extract_item_features(business_df, item_encoder):
    business_df = business_df.copy()
    business_df = business_df[business_df["original_item_id"].isin(item_encoder.classes_)]
    
    # Check if we have any items after filtering
    if len(business_df) == 0:
        print("Warning: No items found after filtering with encoder classes.")
        return torch.zeros((len(item_encoder.classes_), 1))  # Return minimal tensor
    
    business_df["encoded_id"] = item_encoder.transform(business_df["original_item_id"])

    # One-hot encode categories (multi-label)
    business_df["categories"] = business_df["categories"].fillna("").str.lower()
    category_set = set()
    for cats in business_df["categories"]:
        if cats:  # Only process non-empty categories
            category_set.update(c.strip() for c in cats.split(",") if c.strip())
    
    if not category_set:  # If no categories found
        print("Warning: No categories found in business data.")
        return torch.zeros((len(item_encoder.classes_), 1))
    
    category_list = sorted(list(category_set))
    cat_to_idx = {c: i for i, c in enumerate(category_list)}

    item_feature_tensor = torch.zeros((len(item_encoder.classes_), len(cat_to_idx)))
    for _, row in business_df.iterrows():
        encoded_id = int(row["encoded_id"])
        if encoded_id < len(item_encoder.classes_):  # Safety check
            for cat in row["categories"].split(","):
                cat = cat.strip().lower()
                if cat in cat_to_idx:
                    item_feature_tensor[encoded_id][cat_to_idx[cat]] = 1.0

    return item_feature_tensor


def get_pyg_data_objects(edge_index, edge_weight, num_nodes=None):
    """
    Create PyTorch Geometric Data object
    
    Args:
        edge_index: Edge indices tensor
        edge_weight: Edge weights tensor
        num_nodes: Total number of nodes in the graph 
    """
    return Data(
        edge_index=edge_index, 
        edge_weight=edge_weight, 
        y=edge_weight,
        num_nodes=num_nodes
    )

#Datasets/preprocessed(restaurant)
def load_preprocessed_data(save_dir="Datasets\preprocessed"):
    """
    Load preprocessed data from the specified directory.
    """
    print(f"=== Loading preprocessed data from {save_dir} ===")
    
    # Check that files exist
    required_files = ["user_df.csv", "business_df.csv", "review_df.csv",
                     "user_encoder.pkl", "item_encoder.pkl"]
    
    for file in required_files:
        if not os.path.exists(os.path.join(save_dir, file)):
            raise FileNotFoundError(f"File {file} not found in {save_dir}. "
                                  "Run preprocessing first.")
    
    # Load DataFrames
    user_df = pd.read_csv(os.path.join(save_dir, "user_df.csv"))
    business_df = pd.read_csv(os.path.join(save_dir, "business_df.csv"))
    review_df = pd.read_csv(os.path.join(save_dir, "review_df.csv"))
    
    # Load encoders
    with open(os.path.join(save_dir, "user_encoder.pkl"), 'rb') as f:
        user_encoder = pickle.load(f)
    with open(os.path.join(save_dir, "item_encoder.pkl"), 'rb') as f:
        item_encoder = pickle.load(f)
    
    # Extract features with error handling
    print("Extracting user features...")
    user_feats = extract_user_features(user_df, user_encoder)
    print("Extracting item features...")
    item_feats = extract_item_features(business_df, item_encoder)
    
    print(f"✅ Loaded Data: {len(user_df)} users, {len(business_df)} items, {len(review_df)} reviews")
    print(f"User features shape: {user_feats.shape}")
    print(f"Item features shape: {item_feats.shape}")
    
    return user_df, business_df, review_df, user_encoder, item_encoder


def print_dataset_statistics():
    """Print dataset characteristics before main execution"""
    from util.data_utils import load_preprocessed_data
    
    print("=== Dataset Characteristics ===")
    
    # Load the preprocessed data
    user_df, business_df, review_df, user_encoder, item_encoder, _, _ = load_preprocessed_data()
    
    # Calculate statistics
    n_users = user_df['user_id'].nunique()
    n_items = business_df['item_id'].nunique() if 'item_id' in business_df.columns else business_df['original_item_id'].nunique()
    n_ratings = len(review_df)
    
    # Calculate density
    max_possible_ratings = n_users * n_items
    density = (n_ratings / max_possible_ratings) * 100
    
    # Count reviews (assuming reviews have text)
    n_reviews = len(review_df[review_df['text'].notna()]) if 'text' in review_df.columns else n_ratings
    
    # Print the statistics
    print(f"Number of users: {n_users:,}")
    print(f"Number of items: {n_items:,}")
    print(f"Number of ratings: {n_ratings:,}")
    print(f"Number of reviews: {n_reviews:,}")
    print(f"Density: {density:.4f}%")
    print(f"Average ratings per user: {n_ratings/n_users:.2f}")
    print(f"Average ratings per item: {n_ratings/n_items:.2f}")
    print("="*40)
    print()
    
    
def standardize_aspects(aspects: Any) -> List[str]:
    """
    Standardize aspect data to always return a list of cleaned aspect strings.
    This function should be imported and used everywhere aspects are processed.
    """
    # Handle None or NaN
    if aspects is None or (isinstance(aspects, float) and pd.isna(aspects)):
        return []
    
    # If it's a string representation of a list, parse it
    if isinstance(aspects, str):
        if not aspects or aspects == 'None' or aspects == '[]':
            return []
        
        try:
            aspects = ast.literal_eval(aspects)
        except (ValueError, SyntaxError):
            aspects = [aspects]
    
    # Ensure we have a list
    if not isinstance(aspects, list):
        aspects = [aspects] if aspects else []
    
    # Normalize each aspect
    normalized = []
    for aspect in aspects:
        if aspect is not None and aspect != 'None':
            aspect_str = str(aspect)
            cleaned = re.sub(r'^\*\s*', '', aspect_str).strip()
            if cleaned:
                normalized.append(cleaned)
    
    return normalized