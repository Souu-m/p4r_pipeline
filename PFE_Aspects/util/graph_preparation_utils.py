


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
