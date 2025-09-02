


import torch


import torch
from graph_builder import HeterogeneousGraphBuilder
from util.data_utils import load_data, split_data, get_pyg_data_objects, standardize_aspects
from util.data_utils import load_preprocessed_data


import torch
import os
import numpy as np
from graph_builder import HeterogeneousGraphBuilder
from util.data_utils import load_data, split_data, get_pyg_data_objects, standardize_aspects
from util.data_utils import load_preprocessed_data


def save_interaction_splits(train_edges, train_weights, val_edges, val_weights, 
                           test_edges, test_weights, save_dir="Project/Datasets/splits"):
    """Save interaction splits to text files"""
    os.makedirs(save_dir, exist_ok=True)
    
    def save_split(edges, weights, filename):
        filepath = os.path.join(save_dir, filename)
        with open(filepath, 'w') as f:
            for i in range(edges.shape[1]):
                user = edges[0, i].item()
                item = edges[1, i].item()
                weight = weights[i].item()
                f.write(f"{user} {item} {weight}\n")
        print(f"Saved {edges.shape[1]} interactions to {filename}")
    
    save_split(train_edges, train_weights, "train.txt")
    save_split(val_edges, val_weights, "valid.txt")
    save_split(test_edges, test_weights, "test.txt")


def load_interaction_splits(save_dir="Project/Datasets/splits"):
    """Load interaction splits from text files"""
    def load_split(filename):
        filepath = os.path.join(save_dir, filename)
        users, items, weights = [], [], []
        with open(filepath, 'r') as f:
            for line in f:
                u, i, w = line.strip().split()
                users.append(int(u))
                items.append(int(i))
                weights.append(float(w))
        edges = torch.stack([torch.LongTensor(users), torch.LongTensor(items)])
        weights = torch.FloatTensor(weights)
        print(f"Loaded {edges.shape[1]} interactions from {filename}")
        return edges, weights
    
    train_edges, train_weights = load_split("train.txt")
    val_edges, val_weights = load_split("valid.txt")
    test_edges, test_weights = load_split("test.txt")
    
    return (train_edges, train_weights), (val_edges, val_weights), (test_edges, test_weights)


def save_social_edges(edges, weights, filename, save_dir="Project/Datasets/social"):
    """Save social edges (friendship or trust) to file"""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    
    with open(filepath, 'w') as f:
        for i in range(edges.shape[1]):
            user1 = edges[0, i].item()
            user2 = edges[1, i].item()
            weight = weights[i].item()
            f.write(f"{user1} {user2} {weight}\n")
    print(f"Saved {edges.shape[1]} edges to {filepath}")


def load_social_edges(filename, save_dir="Project/Datasets/social"):
    """Load social edges from file"""
    filepath = os.path.join(save_dir, filename)
    
    if not os.path.exists(filepath):
        return None, None
    
    users1, users2, weights = [], [], []
    with open(filepath, 'r') as f:
        for line in f:
            u1, u2, w = line.strip().split()
            users1.append(int(u1))
            users2.append(int(u2))
            weights.append(float(w))
    
    edges = torch.stack([torch.LongTensor(users1), torch.LongTensor(users2)])
    weights = torch.FloatTensor(weights)
    print(f"Loaded {edges.shape[1]} edges from {filepath}")
    return edges, weights


def prepare_data(config, cache_dir="Project/Datasets/cache"):
    """Load data and build graph based on configuration - with caching support"""
    print("=== Loading and preprocessing data ===")
    
    user_df, business_df, review_df, user_encoder, item_encoder = load_preprocessed_data()
    sentiment = False
    if sentiment:
        review_df['sentiment'] = review_df['sentiment'] * 4+1
        alpha = config.ALPHA
        review_df['rating'] = alpha * review_df['sentiment'] + (1-alpha) * review_df['rating']
        
    print("\n=== Building graph ===")
    n_users = user_df['user_id'].nunique()
    n_items = business_df['item_id'].nunique()
    
    builder = HeterogeneousGraphBuilder(n_users, n_items)
    splits_dir = "Project/Datasets/splits"
    ratings_file = os.path.join(splits_dir, "ratings.txt")
    if not all(os.path.exists(os.path.join(splits_dir, f)) 
            for f in ["train.txt", "valid.txt", "test.txt"]):
        print("Creating new splits")
    
        # Always build user-item edges first
        ui_edges, ui_weights = builder.build_user_item_edges(
            review_df, min_rating=config.MIN_RATING, bidirectional=False
        )
        print(f"Built {ui_edges.shape[1]} User-Item edges")
    
        # Save all edges as user-item-rating format to ratings.txt
        print("Saving all edges to ratings.txt")
        with open(ratings_file, 'w') as f:
            for i in range(ui_edges.shape[1]):
                user_id = ui_edges[0, i].item()
                item_id = ui_edges[1, i].item()
                rating = ui_weights[i].item() if ui_weights is not None else 1.0
                f.write(f"{user_id}\t{item_id}\t{rating}\n")
    
    else:
        print("Loading existing edges from ratings.txt")
    
        # Load edges from ratings.txt
        if os.path.exists(ratings_file):
            print("Loading user-item edges from ratings.txt")
        
            users = []
            items = []
            ratings = []
        
            with open(ratings_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        user_id = int(parts[0])
                        item_id = int(parts[1])
                        rating = float(parts[2]) if len(parts) > 2 else 1.0
                    
                        users.append(user_id)
                        items.append(item_id)
                        ratings.append(rating)
        
            # Convert to tensors/arrays (adjust based on your data structure)
            ui_edges = torch.stack([
               torch.LongTensor(users),
               torch.LongTensor(items)
            ])
            ui_weights = torch.FloatTensor(ratings)
            print(f"Loaded {ui_edges.shape[1]} User-Item edges from ratings.txt")
        else:
            print("No ratings.txt found, building edges from scratch")
            ui_edges, ui_weights = builder.build_user_item_edges(
                review_df, min_rating=config.MIN_RATING, bidirectional=False
            )
            print(f"Built {ui_edges.shape[1]} User-Item edges")
    
    social_edges_list = []
    
    # Handle user-user friendship edges with caching
    if config.GRAPH_CONFIG in ['ui_friend', 'ui_friend_trust']:
        # Try to load cached friendship edges
        uu_edges, uu_weights = load_social_edges("friendship.txt")
        
        if uu_edges is None:
            # Build friendship edges if not cached
            print("Building friendship edges (not found in cache)...")
            uu_edges, uu_weights = builder.build_user_user_friendship_edges(
                user_df, user_encoder, min_jaccard=config.MIN_JACCARD
            )
            uu_weights = uu_weights * 4 + 1
            # Save for future use
            save_social_edges(uu_edges, uu_weights, "friendship.txt")
        else:
            print("Using cached friendship edges")
        
        social_edges_list.append((uu_edges, uu_weights))
        print(f"Total friendship edges: {uu_edges.shape[1]}")
    
    # Handle item-item edges
    if config.GRAPH_CONFIG in ['ui_item', 'ui_friend_item', 'ui_trust_item']:
        # Try to load cached item-item edges
        ii_edges, ii_weights = load_social_edges("item_similarity.txt")
        
        if ii_edges is None:
            print("Building item-item edges (not found in cache)...")
            ii_edges, ii_weights = builder.build_item_item_similarity_edges(
                business_df, 
                min_similarity=getattr(config, 'MIN_ITEM_SIMILARITY', 0.6),
                top_k=getattr(config, 'ITEM_TOP_K', 5)
            )
            ii_weights = ii_weights * 4 +1
            # Save for future use
            save_social_edges(ii_edges, ii_weights, "item_similarity.txt")
        else:
            print("Using cached item-item edges")
            
        social_edges_list.append((ii_edges, ii_weights))
        print(f"Total item-item edges: {ii_edges.shape[1]}")
    
    return ui_edges, ui_weights, social_edges_list, n_users, n_items, builder, business_df


def split_and_prepare_data(ui_edges, ui_weights, social_edges_list, builder, config, 
                          n_users, n_items, business_df=None, use_cache=True):
    """Split User-Item interactions, then compute trust from training data only - with caching"""
    
    cache_dir = "Project/Datasets/cache"
    splits_dir = "Project/Datasets/splits"
    
    # Check if splits already exist
    if use_cache and all(os.path.exists(os.path.join(splits_dir, f)) 
                         for f in ["train.txt", "valid.txt", "test.txt"]):
        print("\n=== Loading cached User-Item splits ===")
        (train_ui_e, train_ui_w), (val_ui_e, val_ui_w), (test_ui_e, test_ui_w) = load_interaction_splits()
    else:
        print("\n=== Splitting User-Item interactions ===")
        (train_ui_e, train_ui_w), (val_ui_e, val_ui_w), (test_ui_e, test_ui_w) = split_data(
            ui_edges, ui_weights, train_rate=config.Train_RATE
        )
        # Save splits for future use
        save_interaction_splits(train_ui_e, train_ui_w, val_ui_e, val_ui_w, 
                              test_ui_e, test_ui_w, splits_dir)
    
    print(f"UI Split - Train: {train_ui_e.shape[1]}, Val: {val_ui_e.shape[1]}, Test: {test_ui_e.shape[1]}")
    
    # Handle trust edges with caching
    if config.GRAPH_CONFIG in ['ui_trust', 'ui_friend_trust', 'ui_trust_item']:
        # Try to load cached trust edges
        uu_trust_edges, uu_trust_weights = load_social_edges("trust.txt")
        
        if uu_trust_edges is None:
            print("\n=== Computing trust edges from training data (not found in cache) ===")
            uu_trust_edges, uu_trust_weights = builder.build_user_user_trust_edges(
                train_ui_e, train_ui_w,  # Use TRAINING edges only!
                min_common_items=config.MIN_COMMON_ITEMS,
                min_similarity=config.MIN_SIMILARITY
            )
            uu_trust_weights = uu_trust_weights * 4 + 1
            # Save for future use
            save_social_edges(uu_trust_edges, uu_trust_weights, "trust.txt")
        else:
            print("\n=== Using cached trust edges ===")
        
        print(f"Trust edges: {uu_trust_edges.shape[1]}")
        
        # Handle different configurations
        if config.GRAPH_CONFIG == 'ui_trust':
            social_edges_list = [(uu_trust_edges, uu_trust_weights)]
        
        elif config.GRAPH_CONFIG == 'ui_friend_trust':
            if social_edges_list:  # Should contain friendship edges
                # Try to load cached combined edges
                #combined_edges, combined_weights = load_social_edges("friendship_trust_combined.txt")
                
                    print("Combining friendship and trust edges...")
                    uu_friend_edges, uu_friend_weights = social_edges_list[0]
                    combined_edges, combined_weights = builder.combine_friendship_trust_edges(
                        uu_friend_edges, uu_friend_weights,
                        uu_trust_edges, uu_trust_weights,
                        gamma=config.GAMMA
                    )
                    # Save combined edges
                    social_edges_list = [(combined_edges, combined_weights)]
                    #save_social_edges(combined_edges, combined_weights, "friendship_trust_combined.txt")
            else:
                    print("Use social  friendship please")
                
                #social_edges_list = [(combined_edges, combined_weights)]
            print(f"Combined social edges: {combined_edges.shape[1]}")
        
        elif config.GRAPH_CONFIG == 'ui_trust_item':
            social_edges_list.append((uu_trust_edges, uu_trust_weights))
    
    # Combine all social edges
    if social_edges_list:
        print("Combining all social edges...")
        all_social_edges = []
        all_social_weights = []
        
        for social_e, social_w in social_edges_list:
            if social_e.shape[1] > 0:
                all_social_edges.append(social_e)
                all_social_weights.append(social_w)
        
        if all_social_edges:
            complete_social_edges = torch.cat(all_social_edges, dim=1)
            complete_social_weights = torch.cat(all_social_weights)
            print(f"Total social edges: {complete_social_edges.shape[1]}")
        else:
            complete_social_edges = torch.empty((2, 0), dtype=torch.long)
            complete_social_weights = torch.empty(0, dtype=torch.float)
    else:
        complete_social_edges = torch.empty((2, 0), dtype=torch.long)
        complete_social_weights = torch.empty(0, dtype=torch.float)
        print("No social edges to add")
    
    # Create final datasets
    def create_final_dataset(ui_e, ui_w):
        """Combine UI edges with complete social edges"""
        if complete_social_edges.shape[1] > 0:
            final_edges = torch.cat([ui_e, complete_social_edges], dim=1)
            final_weights = torch.cat([ui_w, complete_social_weights])
        else:
            final_edges, final_weights = ui_e, ui_w
        return final_edges, final_weights
    
    # Create train/val/test datasets
    train_edges, train_weights = create_final_dataset(train_ui_e, train_ui_w)
    val_edges, val_weights = create_final_dataset(val_ui_e, val_ui_w)
    test_edges, test_weights = create_final_dataset(test_ui_e, test_ui_w)
    
    # Create PyG objects
    train_data = get_pyg_data_objects(train_edges, train_weights, num_nodes=n_users + n_items)
    val_data = get_pyg_data_objects(val_edges, val_weights, num_nodes=n_users + n_items)
    test_data = get_pyg_data_objects(test_edges, test_weights, num_nodes=n_users + n_items)
    
    print(f"\nFinal datasets:")
    print(f"  Train: {train_edges.shape[1]} total edges ({train_ui_e.shape[1]} UI + {complete_social_edges.shape[1]} social)")
    print(f"  Val:   {val_edges.shape[1]} total edges ({val_ui_e.shape[1]} UI + {complete_social_edges.shape[1]} social)")
    print(f"  Test:  {test_edges.shape[1]} total edges ({test_ui_e.shape[1]} UI + {complete_social_edges.shape[1]} social)")
    
    return train_data, val_data, test_data


def clear_cache(cache_types=['all']):
    """
    Clear cached files for fresh computation
    
    Args:
        cache_types: List of cache types to clear. Options:
            - 'all': Clear everything
            - 'splits': Clear train/valid/test splits
            - 'friendship': Clear friendship edges
            - 'trust': Clear trust edges
            - 'item': Clear item similarity edges
            - 'combined': Clear combined friendship-trust edges
    """
    if 'all' in cache_types:
        cache_types = ['splits', 'friendship', 'trust', 'item', 'combined']
    
    files_to_delete = []
    
    if 'splits' in cache_types:
        files_to_delete.extend([
            "Project/Datasets/splits/train.txt",
            "Project/Datasets/splits/valid.txt",
            "Project/Datasets/splits/test.txt"
        ])
    
    if 'friendship' in cache_types:
        files_to_delete.append("Project/Datasets/social/friendship.txt")
    
    if 'trust' in cache_types:
        files_to_delete.append("Project/Datasets/social/trust.txt")
    
    if 'item' in cache_types:
        files_to_delete.append("Project/Datasets/social/item_similarity.txt")
    
    if 'combined' in cache_types:
        files_to_delete.append("Project/Datasets/social/friendship_trust_combined.txt")
    
    for filepath in files_to_delete:
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"Deleted: {filepath}")
    
    print("Cache cleared successfully!")








'''
#**********************************************************************************#
def prepare_data(config):
    """Load data and build graph based on configuration - Modified to delay trust computation"""
    print("=== Loading and preprocessing data ===")
   
    user_df, business_df, review_df, user_encoder, item_encoder = load_preprocessed_data()
    sentiment = False
    if sentiment:
        review_df['sentiment'] = review_df['sentiment'] * 4+1
        alpha = config.ALPHA
        review_df['rating'] = alpha * review_df['sentiment'] + (1-alpha) * review_df['rating']
        
    print("\n=== Building graph ===")
    n_users = user_df['user_id'].nunique()
    n_items = business_df['item_id'].nunique()
    
    builder = HeterogeneousGraphBuilder(n_users, n_items)
    
    # Always build user-item edges first
    ui_edges, ui_weights = builder.build_user_item_edges(
        review_df, min_rating=config.MIN_RATING, bidirectional=False
    )
    print(f"Built {ui_edges.shape[1]} User-Item edges")
    
    # For friendship edges - these are safe to compute here (not based on ratings)
    social_edges_list = []
    
    # Handle user-user edges
    if config.GRAPH_CONFIG in ['ui_friend', 'ui_friend_trust']:
    
        uu_edges, uu_weights = builder.build_user_user_friendship_edges(
            user_df, user_encoder, min_jaccard=config.MIN_JACCARD
        )
        uu_weights = uu_weights * 4 + 1
        social_edges_list.append((uu_edges, uu_weights))
        print(f"Built {uu_edges.shape[1]} friendship edges")
    
    # Handle item-item edges
    if config.GRAPH_CONFIG in ['ui_item', 'ui_friend_item', 'ui_trust_item']:
        ii_edges, ii_weights = builder.build_item_item_similarity_edges(
            business_df, 
            min_similarity=getattr(config, 'MIN_ITEM_SIMILARITY', 0.6),
            top_k=getattr(config, 'ITEM_TOP_K', 5)
        )
        ii_weights = ii_weights * 5
        social_edges_list.append((ii_edges, ii_weights))
        print(f"Built {ii_edges.shape[1]} item-item edges")
    
    # For trust edges - we'll compute them AFTER splitting (in split_and_prepare_data)
    # So we don't add them to social_edges_list here
    
    return ui_edges, ui_weights, social_edges_list, n_users, n_items, builder, business_df


def split_and_prepare_data(ui_edges, ui_weights, social_edges_list, builder, config, n_users, n_items, business_df=None):
    """Split User-Item interactions, then compute trust from training data only"""
    print("\n=== Splitting User-Item interactions only ===")
    
    # SPLIT User-Item interactions FIRST
    (train_ui_e, train_ui_w), (val_ui_e, val_ui_w), (test_ui_e, test_ui_w) = split_data(
        ui_edges, ui_weights, train_rate=config.Train_RATE
    )
    
    print(f"UI Split - Train: {train_ui_e.shape[1]}, Val: {val_ui_e.shape[1]}, Test: {test_ui_e.shape[1]}")
    
    # NOW compute trust edges from TRAINING data only
    if config.GRAPH_CONFIG in ['ui_trust', 'ui_friend_trust', 'ui_trust_item']:
        print("\n=== Computing trust edges from training data only ===")
        uu_trust_edges, uu_trust_weights = builder.build_user_user_trust_edges(
            train_ui_e, train_ui_w,  # ← Use TRAINING edges only!
            min_common_items=config.MIN_COMMON_ITEMS,
            min_similarity=config.MIN_SIMILARITY
        )
        uu_trust_weights = uu_trust_weights * 5
        print(f"Built {uu_trust_edges.shape[1]} trust edges from training data")
        
        # Add trust edges to social_edges_list
        if config.GRAPH_CONFIG == 'ui_trust':
            # For ui_trust, trust edges are the only social edges
            social_edges_list = [(uu_trust_edges, uu_trust_weights)]
        
        elif config.GRAPH_CONFIG == 'ui_friend_trust':
            # For ui_friend_trust, we need to combine friendship (already in list) with trust
            if social_edges_list:  # Should contain friendship edges
                uu_friend_edges, uu_friend_weights = social_edges_list[0]
                
                # Combine friendship and trust edges
                uu_combined_edges, uu_combined_weights = builder.combine_friendship_trust_edges(
                    uu_friend_edges, uu_friend_weights,
                    uu_trust_edges, uu_trust_weights,
                    gamma=config.GAMMA
                )
                social_edges_list = [(uu_combined_edges, uu_combined_weights)]
                print(f"Combined into {uu_combined_edges.shape[1]} social edges")
            else:
                # Shouldn't happen, but handle gracefully
                social_edges_list = [(uu_trust_edges, uu_trust_weights)]
        
        elif config.GRAPH_CONFIG == 'ui_trust_item':
            # For ui_trust_item, add trust edges to existing social_edges_list (which should contain item-item edges)
            social_edges_list.append((uu_trust_edges, uu_trust_weights))
    
    # Combine all social edges (now including properly computed trust)
    if social_edges_list:
        print("Combining social edges (kept complete)...")
        all_social_edges = []
        all_social_weights = []
        
        for social_e, social_w in social_edges_list:
            if social_e.shape[1] > 0:
                all_social_edges.append(social_e)
                all_social_weights.append(social_w)
        
        if all_social_edges:
            complete_social_edges = torch.cat(all_social_edges, dim=1)
            complete_social_weights = torch.cat(all_social_weights)
            print(f"Total social edges (complete): {complete_social_edges.shape[1]}")
        else:
            complete_social_edges = torch.empty((2, 0), dtype=torch.long)
            complete_social_weights = torch.empty(0, dtype=torch.float)
    else:
        complete_social_edges = torch.empty((2, 0), dtype=torch.long)
        complete_social_weights = torch.empty(0, dtype=torch.float)
        print("No social edges to add")
    
    # Create final datasets (unchanged)
    def create_final_dataset(ui_e, ui_w):
        """Combine UI edges with complete social edges"""
        if complete_social_edges.shape[1] > 0:
            final_edges = torch.cat([ui_e, complete_social_edges], dim=1)
            final_weights = torch.cat([ui_w, complete_social_weights])
        else:
            final_edges, final_weights = ui_e, ui_w
        return final_edges, final_weights
    
    # Create train/val/test datasets
    train_edges, train_weights = create_final_dataset(train_ui_e, train_ui_w)
    val_edges, val_weights = create_final_dataset(val_ui_e, val_ui_w)
    test_edges, test_weights = create_final_dataset(test_ui_e, test_ui_w)
    
    # Create PyG objects
    train_data = get_pyg_data_objects(train_edges, train_weights, num_nodes=n_users + n_items)
    val_data = get_pyg_data_objects(val_edges, val_weights, num_nodes=n_users + n_items)
    test_data = get_pyg_data_objects(test_edges, test_weights, num_nodes=n_users + n_items)
    
    print(f"Final datasets:")
    print(f"  Train: {train_edges.shape[1]} total edges ({train_ui_e.shape[1]} UI + {complete_social_edges.shape[1]} social)")
    print(f"  Val:   {val_edges.shape[1]} total edges ({val_ui_e.shape[1]} UI + {complete_social_edges.shape[1]} social)")
    print(f"  Test:  {test_edges.shape[1]} total edges ({test_ui_e.shape[1]} UI + {complete_social_edges.shape[1]} social)")
    
    return train_data, val_data, test_data
'''
'''
def prepare_data(config):
    """Load data and build graph based on configuration - Modified to delay trust computation"""
    print("=== Loading and preprocessing data ===")
   
    user_df, business_df, review_df, user_encoder, item_encoder = load_preprocessed_data()
    sentiment= False
    if sentiment:
        review_df['sentiment'] = review_df['sentiment'] * 4+1
        alpha = config.ALPHA
        review_df['rating'] = alpha * review_df['sentiment'] + (1-alpha) * review_df['rating']
        
        
    print("\n=== Building graph ===")
    n_users = user_df['user_id'].nunique()
    n_items = business_df['item_id'].nunique()
    
    builder = HeterogeneousGraphBuilder(n_users, n_items)
    
    # Always build user-item edges first
    ui_edges, ui_weights = builder.build_user_item_edges(
        review_df, min_rating=config.MIN_RATING, bidirectional=False
    )
    print(f"Built {ui_edges.shape[1]} User-Item edges")
    
    # For friendship edges - these are safe to compute here (not based on ratings)
    social_edges_list = []
    
    if config.GRAPH_CONFIG == 'ui_friend':
        uu_edges, uu_weights = builder.build_user_user_friendship_edges(
            user_df, user_encoder, min_jaccard=config.MIN_JACCARD
        )
        uu_weights =  uu_weights * 5
        social_edges_list.append((uu_edges, uu_weights))
        print(f"Built {uu_edges.shape[1]} friendship edges")
    
    # For trust edges - we'll compute them AFTER splitting (in split_and_prepare_data)
    # So we don't add them to social_edges_list here
    elif config.GRAPH_CONFIG in ['ui_trust', 'ui_friend_trust']:
        # We'll handle trust edges after splitting
        # For ui_friend_trust, we still compute friendship here
        if config.GRAPH_CONFIG == 'ui_friend_trust':
            uu_edges, uu_weights = builder.build_user_user_friendship_edges(
                user_df, user_encoder, min_jaccard=config.MIN_JACCARD
            )
            uu_weights = uu_weights * 5
            social_edges_list.append((uu_edges, uu_weights))
            print(f"Built {uu_edges.shape[1]} friendship edges")
    
    return ui_edges, ui_weights, social_edges_list, n_users, n_items, builder


def split_and_prepare_data(ui_edges, ui_weights, social_edges_list, builder, config, n_users, n_items):
    """Split User-Item interactions, then compute trust from training data only"""
    print("\n=== Splitting User-Item interactions only ===")
    
    # SPLIT User-Item interactions FIRST
    (train_ui_e, train_ui_w), (val_ui_e, val_ui_w), (test_ui_e, test_ui_w) = split_data(
        ui_edges, ui_weights, train_rate=config.Train_RATE
    )
    
    print(f"UI Split - Train: {train_ui_e.shape[1]}, Val: {val_ui_e.shape[1]}, Test: {test_ui_e.shape[1]}")
    
    # NOW compute trust edges from TRAINING data only
    if config.GRAPH_CONFIG in ['ui_trust', 'ui_friend_trust']:
        print("\n=== Computing trust edges from training data only ===")
        uu_trust_edges, uu_trust_weights = builder.build_user_user_trust_edges(
            train_ui_e, train_ui_w,  # ← Use TRAINING edges only!
            min_common_items=config.MIN_COMMON_ITEMS,
            min_similarity=config.MIN_SIMILARITY
        )
        uu_trust_weights =  uu_trust_weights * 5
        print(f"Built {uu_trust_edges.shape[1]} trust edges from training data")
        
        # Add trust edges to social_edges_list
        if config.GRAPH_CONFIG == 'ui_trust':
            # For ui_trust, trust edges are the only social edges
            social_edges_list = [(uu_trust_edges, uu_trust_weights)]
        
        elif config.GRAPH_CONFIG == 'ui_friend_trust':
            # For ui_friend_trust, we need to combine friendship (already in list) with trust
            if social_edges_list:  # Should contain friendship edges
                uu_friend_edges, uu_friend_weights = social_edges_list[0]
                
                # Combine friendship and trust edges
                uu_combined_edges, uu_combined_weights = builder.combine_friendship_trust_edges(
                    uu_friend_edges, uu_friend_weights,
                    uu_trust_edges, uu_trust_weights,
                    gamma=config.GAMMA
                )
                social_edges_list = [(uu_combined_edges, uu_combined_weights)]
                print(f"Combined into {uu_combined_edges.shape[1]} social edges")
            else:
                # Shouldn't happen, but handle gracefully
                social_edges_list = [(uu_trust_edges, uu_trust_weights)]
    
    # Combine all social edges (now including properly computed trust)
    if social_edges_list:
        print("Combining social edges (kept complete)...")
        all_social_edges = []
        all_social_weights = []
        
        for social_e, social_w in social_edges_list:
            if social_e.shape[1] > 0:
                all_social_edges.append(social_e)
                all_social_weights.append(social_w)
        
        if all_social_edges:
            complete_social_edges = torch.cat(all_social_edges, dim=1)
            complete_social_weights = torch.cat(all_social_weights)
            print(f"Total social edges (complete): {complete_social_edges.shape[1]}")
        else:
            complete_social_edges = torch.empty((2, 0), dtype=torch.long)
            complete_social_weights = torch.empty(0, dtype=torch.float)
    else:
        complete_social_edges = torch.empty((2, 0), dtype=torch.long)
        complete_social_weights = torch.empty(0, dtype=torch.float)
        print("No social edges to add")
    
    # Create final datasets (unchanged)
    def create_final_dataset(ui_e, ui_w):
        """Combine UI edges with complete social edges"""
        if complete_social_edges.shape[1] > 0:
            final_edges = torch.cat([ui_e, complete_social_edges], dim=1)
            final_weights = torch.cat([ui_w, complete_social_weights])
        else:
            final_edges, final_weights = ui_e, ui_w
        return final_edges, final_weights
    
    # Create train/val/test datasets
    train_edges, train_weights = create_final_dataset(train_ui_e, train_ui_w)
    val_edges, val_weights = create_final_dataset(val_ui_e, val_ui_w)
    test_edges, test_weights = create_final_dataset(test_ui_e, test_ui_w)
    
    # Create PyG objects
    train_data = get_pyg_data_objects(train_edges, train_weights, num_nodes=n_users + n_items)
    val_data = get_pyg_data_objects(val_edges, val_weights, num_nodes=n_users + n_items)
    test_data = get_pyg_data_objects(test_edges, test_weights, num_nodes=n_users + n_items)
    
    print(f"Final datasets:")
    print(f"  Train: {train_edges.shape[1]} total edges ({train_ui_e.shape[1]} UI + {complete_social_edges.shape[1]} social)")
    print(f"  Val:   {val_edges.shape[1]} total edges ({val_ui_e.shape[1]} UI + {complete_social_edges.shape[1]} social)")
    print(f"  Test:  {test_edges.shape[1]} total edges ({test_ui_e.shape[1]} UI + {complete_social_edges.shape[1]} social)")
    
    return train_data, val_data, test_data

'''