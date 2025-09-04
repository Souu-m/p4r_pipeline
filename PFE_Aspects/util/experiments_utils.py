
import glob
import json
# Create experiment folder
import os
import random
from datetime import datetime

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from graph_builder import HeterogeneousGraphBuilder
from models.models_clean import SENGR_GCN
from preprocessing import Preprocessor
from torch_geometric.loader import DataLoader
from util.data_utils import (get_pyg_data_objects, load_data,
                             load_preprocessed_data, split_data)
from util.evaluation_utils import evaluate_model, print_metrics
from util.graph_preparation_utils import prepare_data, split_and_prepare_data
from util.train_utils import train_model

from main_global import set_seed
from main_global import train_and_evaluate,main_global

import copy
from datetime import datetime


def exp_GCN(config):
    # Define graph configurations to test
    graph_configs = ['ui_friend','ui_trust']  # Order matters for consistency
    
    # Grid search parameters
    embedding_dims = [16,32,64]
    num_layers_list = [1,4]
    
    # Store results for all configurations
    all_results = {}
    
    # Iterate over each graph configuration
    for graph_config in graph_configs:
        print(f"\n{'='*50}")
        print(f"Testing Graph Configuration: {graph_config}")
        print(f"{'='*50}\n")
        # Prepare data
        config.GRAPH_CONFIG=graph_config
        #edge_list, n_users, n_items, builder = prepare_data(config)
        #train_data, val_data, test_data = split_and_prepare_data(edge_list, builder, config)
        ui_edges, ui_weights, social_edges_list, n_users, n_items, builder = prepare_data(config)
        train_data, val_data, test_data = split_and_prepare_data(ui_edges,ui_weights,social_edges_list ,builder, config, n_users, n_items)
        results = []
        
        # Test each combination for current graph config
        for emb_dim in embedding_dims:
            for n_layers in num_layers_list:
                print(f"\n=== Testing {graph_config}: emb_dim={emb_dim}, layers={n_layers} ===")
                
                # Create config
                # Config.USE_TRANSFORMER = False
                config.EMBEDDING_DIM = emb_dim
                config.NUM_LAYERS = n_layers
                
                """Run single experiment with config settings"""
                print("Graph config:", config.GRAPH_CONFIG)
                print("Embedding size:", config.EMBEDDING_DIM)
                print("Number of layers:", config.NUM_LAYERS)
                print("Gamma:", config.GAMMA)
                

                # Train and evaluate
                model, test_metrics = train_and_evaluate(
                    config, train_data, val_data, test_data, n_users, n_items
                )
                
                print_metrics(test_metrics, k=config.k)
                print(f"RMSE: {test_metrics['rmse']:.4f}, MAE: {test_metrics['mae']:.4f}")
                
                results.append({
                    'graph_config': graph_config,
                    'emb_dim': emb_dim,
                    'num_layers': n_layers,
                    'metrics': test_metrics
                })
        
        # Store results for this graph config
        all_results[graph_config] = results
        
        # Find best for this graph config
        best_result = min(results, key=lambda x: x['metrics']['rmse'])
        print(f"\n=== Best for {graph_config}: emb_dim={best_result['emb_dim']}, "
              f"layers={best_result['num_layers']}, RMSE={best_result['metrics']['rmse']:.4f} ===")
        
        # Save results for this specific graph configuration
        filename = f'grid_search_results_{graph_config}.json'
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filename}")
    
    # Find overall best configuration across all graph types
    print(f"\n{'='*50}")
    print("OVERALL BEST CONFIGURATION")
    print(f"{'='*50}")
    
    overall_best = None
    overall_best_rmse = float('inf')
    
    for graph_config, results in all_results.items():
        best_in_config = min(results, key=lambda x: x['metrics']['rmse'])
        if best_in_config['metrics']['rmse'] < overall_best_rmse:
            overall_best_rmse = best_in_config['metrics']['rmse']
            overall_best = best_in_config
    
    print(f"Best overall: Graph={overall_best['graph_config']}, "
          f"emb_dim={overall_best['emb_dim']}, layers={overall_best['num_layers']}, "
          f"RMSE={overall_best['metrics']['rmse']:.4f}")
    
    # Save all results in one comprehensive file
    with open('all_grid_search_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save summary of best configurations
    summary = {
        'best_per_graph': {},
        'overall_best': overall_best
    }
    
    for graph_config, results in all_results.items():
        best = min(results, key=lambda x: x['metrics']['rmse'])
        summary['best_per_graph'][graph_config] = {
            'emb_dim': best['emb_dim'],
            'num_layers': best['num_layers'],
            'rmse': best['metrics']['rmse'],
            'mae': best['metrics']['mae']
        }
    
    with open('grid_search_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nAll results saved!")
    print("- Individual graph results: grid_search_results_[graph_config].json")
    print("- All results: all_grid_search_results.json")
    print("- Summary: grid_search_summary.json")




def exp_GCN_up_1(config):

    
    # Define graph configurations to test
    graph_configs = ['ui','ui_friend','ui_trust']  # Order matters for consistency
    seed=config.SEED

    # Grid search parameters
    embedding_dims = [16,32,64,128]
    num_layers_list = [1, 4]
    transformer_layers = [1, 2, 3, 4]
    transformer_heads = [1, 2, 4]
    
    # Store results for all configurations
    all_results = {}
    
    # Add timestamp to avoid overwriting previous results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Iterate over each graph configuration
    for graph_config in graph_configs:
        set_seed(seed)

        print(f"\n{'='*50}")
        print(f"Testing Graph Configuration: {graph_config}")
        print(f"{'='*50}\n")
        # Create a copy of config to avoid mutation
        exp_config = copy.deepcopy(config)
        exp_config.GRAPH_CONFIG = graph_config
        
        # Prepare data for this graph configuration
        ui_edges, ui_weights, social_edges_list, n_users, n_items, builder = prepare_data(exp_config)
        train_data, val_data, test_data = split_and_prepare_data(
            ui_edges, ui_weights, social_edges_list, builder, exp_config, n_users, n_items
        )
        
        results = []
        
        # Test each combination for current graph config
        for emb_dim in embedding_dims:
            for n_layers in num_layers_list:
                for trans_layers in transformer_layers:
                    for trans_heads in transformer_heads:
                        set_seed(seed)

                        print(f"\n=== Testing {graph_config}: emb_dim={emb_dim}, layers={n_layers}, trans_layers={trans_layers}, trans_heads={trans_heads} ===")
                        
                        # Create experiment-specific config
                        trial_config = copy.deepcopy(exp_config)
                        trial_config.EMBEDDING_DIM = emb_dim
                        trial_config.NUM_LAYERS = n_layers
                        trial_config.TRANSFORMER_LAYERS = trans_layers
                        trial_config.TRANSFORMER_HEADS = trans_heads
                        
                        # Ensure transformer is on for these experiments
                        trial_config.USE_TRANSFORMER = True
                        trial_config.USE_EDGE_FEATURES = True
                        
                        try:
                            # Train and evaluate
                            model, test_metrics = train_and_evaluate(
                                trial_config, train_data, val_data, test_data, n_users, n_items
                            )
                            
                            print_metrics(test_metrics, k=trial_config.k)
                            print(f"RMSE: {test_metrics['rmse']:.4f}, MAE: {test_metrics['mae']:.4f}")
                            
                            # Store additional info for analysis
                            results.append({
                                'graph_config': graph_config,
                                'emb_dim': emb_dim,
                                'num_layers': n_layers,
                                'transformer_layers': trans_layers,
                                'transformer_heads': trans_heads,
                                'metrics': test_metrics,
                                'training_time': test_metrics.get('training_time', None)
                            })
                            
                        except Exception as e:
                            print(f"Error in configuration: {e}")
                            results.append({
                                'graph_config': graph_config,
                                'emb_dim': emb_dim,
                                'num_layers': n_layers,
                                'transformer_layers': trans_layers,
                                'transformer_heads': trans_heads,
                                'metrics': {'rmse': float('inf'), 'mae': float('inf')},
                                'error': str(e)
                            })
        
        # Store results for this graph config
        all_results[graph_config] = results
        
        # Find best for this graph config (excluding failed runs)
        valid_results = [r for r in results if 'error' not in r]
        if valid_results:
            best_result = min(valid_results, key=lambda x: x['metrics']['rmse'])
            print(f"\n=== Best for {graph_config}: emb_dim={best_result['emb_dim']}, "
                  f"layers={best_result['num_layers']}, trans_layers={best_result['transformer_layers']}, "
                  f"trans_heads={best_result['transformer_heads']}, RMSE={best_result['metrics']['rmse']:.4f} ===")
        
        # Save results with timestamp
        filename = f'grid_search_results_{graph_config}_sim_{config.MIN_SIMILARITY}_jacc_{config.MIN_JACCARD}_{timestamp}.json'
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filename}")
    
    # Find overall best configuration
    print(f"\n{'='*50}")
    print("OVERALL BEST CONFIGURATION")
    print(f"{'='*50}")
    
    overall_best = None
    overall_best_rmse = float('inf')
    
    for graph_config, results in all_results.items():
        valid_results = [r for r in results if 'error' not in r]
        if valid_results:
            best_in_config = min(valid_results, key=lambda x: x['metrics']['rmse'])
            if best_in_config['metrics']['rmse'] < overall_best_rmse:
                overall_best_rmse = best_in_config['metrics']['rmse']
                overall_best = best_in_config
    
    if overall_best:
        print(f"Best overall: Graph={overall_best['graph_config']}, "
              f"emb_dim={overall_best['emb_dim']}, layers={overall_best['num_layers']}, "
              f"trans_layers={overall_best['transformer_layers']}, trans_heads={overall_best['transformer_heads']}, "
              f"RMSE={overall_best['metrics']['rmse']:.4f}")
    
    # Save all results with timestamp
    with open(f'all_grid_search_results_{timestamp}.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create analysis-friendly summary
    summary = {
        'timestamp': timestamp,
        'total_experiments': sum(len(results) for results in all_results.values()),
        'graph_configs_tested': list(all_results.keys()),
        'embedding_dims_tested': embedding_dims,
        'num_layers_tested': num_layers_list,
        'transformer_layers_tested': transformer_layers,
        'transformer_heads_tested': transformer_heads,
        'best_per_graph': {},
        'overall_best': overall_best
    }
    
    for graph_config, results in all_results.items():
        valid_results = [r for r in results if 'error' not in r]
        if valid_results:
            best = min(valid_results, key=lambda x: x['metrics']['rmse'])
            summary['best_per_graph'][graph_config] = {
                'emb_dim': best['emb_dim'],
                'num_layers': best['num_layers'],
                'transformer_layers': best['transformer_layers'],
                'transformer_heads': best['transformer_heads'],
                'rmse': best['metrics']['rmse'],
                'mae': best['metrics']['mae'],
                'all_metrics': best['metrics']  # Store all metrics for analysis
            }
    
    with open(f'grid_search_summary_{timestamp}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nAll results saved!")
    print(f"- Individual graph results: grid_search_results_[graph_config]_{timestamp}.json")
    print(f"- All results: all_grid_search_results_{timestamp}.json")
    print(f"- Summary: grid_search_summary_{timestamp}.json")
    print(f"min similairy for trust edges is set to {config.MIN_SIMILARITY} and min items for trust edges is set to {config.MIN_COMMON_ITEMS}")
    
    return summary  # Return summary for further processing
    
    
def exp_GCN_up(config):

    
    # Define graph configurations to test
    graph_configs = ['ui','ui_friend','ui_trust']  # Order matters for consistency
    seed=config.SEED

    # Grid search parameters
    embedding_dims = [16,32,64,128,256]
    num_layers_list = [1,2,3,4]
    
    # Store results for all configurations
    all_results = {}
    
    # Add timestamp to avoid overwriting previous results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Iterate over each graph configuration
    for graph_config in graph_configs:

        print(f"\n{'='*50}")
        print(f"Testing Graph Configuration: {graph_config}")
        print(f"{'='*50}\n")
        # Create a copy of config to avoid mutation
        exp_config = copy.deepcopy(config)
        exp_config.GRAPH_CONFIG = graph_config
        
        # Prepare data for this graph configuration
        ui_edges, ui_weights, social_edges_list, n_users, n_items, builder ,business_df= prepare_data(exp_config)
        train_data, val_data, test_data = split_and_prepare_data(
            ui_edges, ui_weights, social_edges_list, builder, exp_config, n_users, n_items,business_df
        )
        #ui_edges, ui_weights, social_edges_list, n_users, n_items, builder,review_df = prepare_data(config)
        #train_data, val_data, test_data = split_and_prepare_data(ui_edges,ui_weights,social_edges_list ,builder, config,n_users, n_items,review_df)
        
        results = []
        
        # Test each combination for current graph config
        for emb_dim in embedding_dims:

            for n_layers in num_layers_list:
  

                print(f"\n=== Testing {graph_config}: emb_dim={emb_dim}, layers={n_layers} ===")
                
                # Create experiment-specific config
                trial_config = copy.deepcopy(exp_config)
                trial_config.EMBEDDING_DIM = emb_dim
                trial_config.NUM_LAYERS = n_layers
                
                # Ensure transformer is off for base GCN experiments
                trial_config.USE_TRANSFORMER = False
                trial_config.USE_EDGE_FEATURES = True
                
                try:
                    # Train and evaluate
                    model, test_metrics = train_and_evaluate(
                        trial_config, train_data, val_data, test_data, n_users, n_items
                    )
                    
                    print_metrics(test_metrics, k=trial_config.k)
                    print(f"RMSE: {test_metrics['rmse']:.4f}, MAE: {test_metrics['mae']:.4f}")
                    
                    # Store additional info for analysis
                    results.append({
                        'graph_config': graph_config,
                        'emb_dim': emb_dim,
                        'num_layers': n_layers,
                        'metrics': test_metrics,
                        'training_time': test_metrics.get('training_time', None)
                    })
                    
                except Exception as e:
                    print(f"Error in configuration: {e}")
                    results.append({
                        'graph_config': graph_config,
                        'emb_dim': emb_dim,
                        'num_layers': n_layers,
                        'metrics': {'rmse': float('inf'), 'mae': float('inf')},
                        'error': str(e)
                    })
        
        # Store results for this graph config
        all_results[graph_config] = results
        
        # Find best for this graph config (excluding failed runs)
        valid_results = [r for r in results if 'error' not in r]
        if valid_results:
            best_result = min(valid_results, key=lambda x: x['metrics']['rmse'])
            print(f"\n=== Best for {graph_config}: emb_dim={best_result['emb_dim']}, "
                  f"layers={best_result['num_layers']}, RMSE={best_result['metrics']['rmse']:.4f} ===")
        
        # Save results with timestamp
        filename = f'grid_search_results_{graph_config}_sim_{config.MIN_SIMILARITY}_jacc_{config.MIN_JACCARD}_{timestamp}.json'
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filename}")
    
    # Find overall best configuration
    print(f"\n{'='*50}")
    print("OVERALL BEST CONFIGURATION")
    print(f"{'='*50}")
    
    overall_best = None
    overall_best_rmse = float('inf')
    
    for graph_config, results in all_results.items():
        valid_results = [r for r in results if 'error' not in r]
        if valid_results:
            best_in_config = min(valid_results, key=lambda x: x['metrics']['rmse'])
            if best_in_config['metrics']['rmse'] < overall_best_rmse:
                overall_best_rmse = best_in_config['metrics']['rmse']
                overall_best = best_in_config
    
    if overall_best:
        print(f"Best overall: Graph={overall_best['graph_config']}, "
              f"emb_dim={overall_best['emb_dim']}, layers={overall_best['num_layers']}, "
              f"RMSE={overall_best['metrics']['rmse']:.4f}")
    
    # Save all results with timestamp
    with open(f'all_grid_search_results_{timestamp}.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create analysis-friendly summary
    summary = {
        'timestamp': timestamp,
        'total_experiments': sum(len(results) for results in all_results.values()),
        'graph_configs_tested': list(all_results.keys()),
        'embedding_dims_tested': embedding_dims,
        'num_layers_tested': num_layers_list,
        'best_per_graph': {},
        'overall_best': overall_best
    }
    
    for graph_config, results in all_results.items():
        valid_results = [r for r in results if 'error' not in r]
        if valid_results:
            best = min(valid_results, key=lambda x: x['metrics']['rmse'])
            summary['best_per_graph'][graph_config] = {
                'emb_dim': best['emb_dim'],
                'num_layers': best['num_layers'],
                'rmse': best['metrics']['rmse'],
                'mae': best['metrics']['mae'],
                'all_metrics': best['metrics']  # Store all metrics for analysis
            }
    
    with open(f'grid_search_summary_{timestamp}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nAll results saved!")
    print(f"- Individual graph results: grid_search_results_[graph_config]_{timestamp}.json")
    print(f"- All results: all_grid_search_results_{timestamp}.json")
    print(f"- Summary: grid_search_summary_{timestamp}.json")
    print(f"min similairy for trust edges is set to {config.MIN_SIMILARITY} and min items for trust edges is set to {config.MIN_COMMON_ITEMS}")
    
    return summary  # Return summary for further processing
    
    
    
def exp_GCN_up_indices(config):
    import copy
    from datetime import datetime
    from itertools import product
    
    # Define experiment parameters
    graph_configs = ['ui', 'ui_friend', 'ui_trust']  
    embedding_dims = [16, 32, 64, 128, 192, 256]
    num_layers_list = [1, 2, 3, 4]
    
    # Similarity thresholds for different edge types
    jaccard_thresholds = [0.1, 0.3, 0.5, 0.7]  # For friendship edges
    cosine_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]  # For trust edges
    
    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for graph_config in graph_configs:
        print(f"\n{'='*50}")
        print(f"Testing Graph Configuration: {graph_config}")
        print(f"{'='*50}\n")
        
        results = []
        
        # Determine which thresholds to iterate based on graph type
        if graph_config == 'ui':
            # No social edges, just test different embeddings/layers
            threshold_combinations = [(None, None)]
        elif graph_config == 'ui_friend':
            # Only Jaccard threshold matters
            threshold_combinations = [(j, None) for j in jaccard_thresholds]
        elif graph_config == 'ui_trust':
            # Only cosine similarity threshold matters
            threshold_combinations = [(None, c) for c in cosine_thresholds]
        
        # Create all combinations
        all_combinations = list(product(embedding_dims, num_layers_list, threshold_combinations))
        
        for emb_dim, n_layers, (jaccard_thresh, cosine_thresh) in all_combinations:
            # Create config for this experiment
            trial_config = copy.deepcopy(config)
            trial_config.GRAPH_CONFIG = graph_config
            trial_config.EMBEDDING_DIM = emb_dim
            trial_config.NUM_LAYERS = n_layers
            trial_config.USE_TRANSFORMER = False
            trial_config.USE_EDGE_FEATURES = False
            
            # Set thresholds based on graph type
            if jaccard_thresh is not None:
                trial_config.MIN_JACCARD = jaccard_thresh
            if cosine_thresh is not None:
                trial_config.MIN_SIMILARITY = cosine_thresh
            
            # Print current configuration
            thresh_info = ""
            if graph_config == 'ui_friend':
                thresh_info = f", jaccard={jaccard_thresh}"
            elif graph_config == 'ui_trust':
                thresh_info = f", cosine={cosine_thresh}"
            
            print(f"\n=== Testing {graph_config}: emb_dim={emb_dim}, layers={n_layers}{thresh_info} ===")
            
            # Prepare data with current thresholds
            ui_edges, ui_weights, social_edges_list, n_users, n_items, builder = prepare_data(trial_config)
            train_data, val_data, test_data = split_and_prepare_data(
                ui_edges, ui_weights, social_edges_list, builder, trial_config, n_users, n_items
            )
            
            try:
                model, test_metrics = train_and_evaluate(
                    trial_config, train_data, val_data, test_data, n_users, n_items
                )
                
                print_metrics(test_metrics, k=trial_config.k)
                print(f"RMSE: {test_metrics['rmse']:.4f}, MAE: {test_metrics['mae']:.4f}")
                
                result_dict = {
                    'graph_config': graph_config,
                    'emb_dim': emb_dim,
                    'num_layers': n_layers,
                    'metrics': test_metrics
                }
                
                # Add threshold info based on graph type
                if graph_config == 'ui_friend':
                    result_dict['min_jaccard'] = jaccard_thresh
                elif graph_config == 'ui_trust':
                    result_dict['min_similarity'] = cosine_thresh
                
                results.append(result_dict)
                
            except Exception as e:
                print(f"Error: {e}")
                result_dict = {
                    'graph_config': graph_config,
                    'emb_dim': emb_dim,
                    'num_layers': n_layers,
                    'metrics': {'rmse': float('inf'), 'mae': float('inf')},
                    'error': str(e)
                }
                if graph_config == 'ui_friend':
                    result_dict['min_jaccard'] = jaccard_thresh
                elif graph_config == 'ui_trust':
                    result_dict['min_similarity'] = cosine_thresh
                
                results.append(result_dict)
        
        all_results[graph_config] = results
        
        # Find best for this graph config
        valid_results = [r for r in results if 'error' not in r]
        if valid_results:
            best_result = min(valid_results, key=lambda x: x['metrics']['rmse'])
            print(f"\n=== Best for {graph_config}: ")
            print(f"    emb_dim={best_result['emb_dim']}, layers={best_result['num_layers']}")
            if graph_config == 'ui_friend':
                print(f"    min_jaccard={best_result['min_jaccard']}")
            elif graph_config == 'ui_trust':
                print(f"    min_similarity={best_result['min_similarity']}")
            print(f"    RMSE={best_result['metrics']['rmse']:.4f} ===")
        
        # Save results
        with open(f'grid_search_results_{graph_config}_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    # Find overall best
    print(f"\n{'='*50}")
    print("OVERALL BEST CONFIGURATION")
    print(f"{'='*50}")
    
    overall_best = min(
        (r for results in all_results.values() for r in results if 'error' not in r),
        key=lambda x: x['metrics']['rmse'],
        default=None
    )
    
    if overall_best:
        print(f"Best overall: Graph={overall_best['graph_config']}, "
              f"emb_dim={overall_best['emb_dim']}, layers={overall_best['num_layers']}, "
              f"RMSE={overall_best['metrics']['rmse']:.4f}")
    
    # Save all results and summary
    with open(f'all_grid_search_results_{timestamp}.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    summary = {
        'timestamp': timestamp,
        'total_experiments': sum(len(results) for results in all_results.values()),
        'best_per_graph': {},
        'overall_best': overall_best
    }
    
    for graph_config, results in all_results.items():
        valid_results = [r for r in results if 'error' not in r]
        if valid_results:
            best = min(valid_results, key=lambda x: x['metrics']['rmse'])
            summary['best_per_graph'][graph_config] = best
    
    with open(f'grid_search_summary_{timestamp}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved with timestamp: {timestamp}")
    
    return summary






def exp_trans_up(config):
    """
    Run Transformer experiments based on best GCN configurations.
    Tests transformer layers and heads with/without edge features.
    """
    import copy
    from datetime import datetime
    import json
    
    # Define configurations to test (based on GCN best results)
    configs_to_test = [
        {'GRAPH_CONFIG': 'ui', 'USE_TRANSFORMER': True, 'USE_EDGE_FEATURES': False, 
         'EMBEDDING_DIM': 32, 'NUM_LAYERS': 4},
        {'GRAPH_CONFIG': 'ui', 'USE_TRANSFORMER': True, 'USE_EDGE_FEATURES': True, 
         'EMBEDDING_DIM': 32, 'NUM_LAYERS': 4},
        {'GRAPH_CONFIG': 'ui_friend_trust', 'USE_TRANSFORMER': True, 'USE_EDGE_FEATURES': False, 
         'EMBEDDING_DIM': 32, 'NUM_LAYERS': 1, 'GAMMA': 0.3},
        {'GRAPH_CONFIG': 'ui_friend_trust', 'USE_TRANSFORMER': True, 'USE_EDGE_FEATURES': True, 
          'EMBEDDING_DIM': 32, 'NUM_LAYERS': 1, 'GAMMA': 0.3}
    ]
    
    transformer_layers = [1, 2, 3, 4]
    transformer_heads = [1, 2, 4,8]
    
    seed = config.SEED
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Store all results
    all_results = []
    
    # Cache prepared data for each graph config to avoid redundant preparation
    data_cache = {}
    
    print(f"\n{'='*70}")
    print(f"TRANSFORMER EXPERIMENTS")
    print(f"Testing {len(configs_to_test)} configs × {len(transformer_layers)} layers × {len(transformer_heads)} heads")
    print(f"{'='*70}\n")
    
    for config_idx, test_config in enumerate(configs_to_test):
        graph_config = test_config['GRAPH_CONFIG']
        edge_features = test_config['USE_EDGE_FEATURES']
        
        print(f"\n{'='*60}")
        print(f"Config {config_idx+1}/{len(configs_to_test)}: {graph_config}, Edge Features: {edge_features}")
        print(f"{'='*60}\n")
        
        # Prepare data once per unique graph configuration
        if graph_config not in data_cache:
            set_seed(seed)
            
            # Create config for data preparation
            data_config = copy.deepcopy(config)
            data_config.GRAPH_CONFIG = graph_config
            if 'GAMMA' in test_config:
                data_config.GAMMA = test_config['GAMMA']
            
            # Prepare data
            ui_edges, ui_weights, social_edges_list, n_users, n_items, builder = prepare_data(data_config)
            train_data, val_data, test_data = split_and_prepare_data(
                ui_edges, ui_weights, social_edges_list, builder, data_config, n_users, n_items
            )
            
            data_cache[graph_config] = {
                'train_data': train_data,
                'val_data': val_data,
                'test_data': test_data,
                'n_users': n_users,
                'n_items': n_items
            }
            print(f"Data prepared and cached for {graph_config}")
        
        # Get cached data
        cached = data_cache[graph_config]
        train_data = cached['train_data']
        val_data = cached['val_data']
        test_data = cached['test_data']
        n_users = cached['n_users']
        n_items = cached['n_items']
        
        # Test each transformer configuration
        for t_layers in transformer_layers:
            for t_heads in transformer_heads:
                set_seed(seed)
                
                # Create experiment config
                exp_config = copy.deepcopy(config)
                
                # Apply all settings from test_config
                for key, value in test_config.items():
                    setattr(exp_config, key, value)
                
                # Set transformer-specific parameters
                exp_config.TRANSFORMER_LAYERS = t_layers
                exp_config.TRANSFORMER_HEADS = t_heads
                
                print(f"\n=== Testing: {graph_config}, Edge={edge_features}, "
                      f"T-Layers={t_layers}, T-Heads={t_heads} ===")
                
                try:
                    # Train and evaluate
                    model, test_metrics = train_and_evaluate(
                        exp_config, train_data, val_data, test_data, n_users, n_items,
                        embedding_dim=exp_config.EMBEDDING_DIM,
                        num_layers=exp_config.NUM_LAYERS,
                        transformer_layers=t_layers,
                        transformer_heads=t_heads
                    )
                    
                    print_metrics(test_metrics, k=exp_config.k)
                    print(f"RMSE: {test_metrics['rmse']:.4f}, MAE: {test_metrics['mae']:.4f}")
                    
                    # Store results
                    result = {
                        'graph_config': graph_config,
                        'use_edge_features': edge_features,
                        'embedding_dim': test_config['EMBEDDING_DIM'],
                        'num_layers': test_config['NUM_LAYERS'],
                        'transformer_layers': t_layers,
                        'transformer_heads': t_heads,
                        'metrics': test_metrics,
                        'gamma': test_config.get('GAMMA', None)
                    }
                    all_results.append(result)
                    
                except Exception as e:
                    print(f"Error in configuration: {e}")
                    all_results.append({
                        'graph_config': graph_config,
                        'use_edge_features': edge_features,
                        'embedding_dim': test_config['EMBEDDING_DIM'],
                        'num_layers': test_config['NUM_LAYERS'],
                        'transformer_layers': t_layers,
                        'transformer_heads': t_heads,
                        'metrics': {'rmse': float('inf'), 'mae': float('inf')},
                        'error': str(e)
                    })
    
    # Find best configurations
    print(f"\n{'='*70}")
    print("BEST TRANSFORMER CONFIGURATIONS")
    print(f"{'='*70}\n")
    
    # Group results by graph config and edge features
    from collections import defaultdict
    grouped_results = defaultdict(list)
    
    for result in all_results:
        if 'error' not in result:
            key = (result['graph_config'], result['use_edge_features'])
            grouped_results[key].append(result)
    
    best_per_config = {}
    for (graph_config, use_edges), results in grouped_results.items():
        if results:
            best = min(results, key=lambda x: x['metrics']['rmse'])
            best_per_config[(graph_config, use_edges)] = best
            
            print(f"{graph_config} (Edge Features={use_edges}):")
            print(f"  Best: T-Layers={best['transformer_layers']}, T-Heads={best['transformer_heads']}")
            print(f"  Embedding Dim={best['embedding_dim']}, Num Layers={best['num_layers']}")
            print(f"  RMSE={best['metrics']['rmse']:.4f}, MAE={best['metrics']['mae']:.4f}")
    
    # Find overall best
    if best_per_config:
        overall_best_key = min(best_per_config.keys(), 
                               key=lambda k: best_per_config[k]['metrics']['rmse'])
        overall_best = best_per_config[overall_best_key]
        
        print(f"\n{'='*50}")
        print("OVERALL BEST TRANSFORMER CONFIG")
        print(f"{'='*50}")
        print(f"Graph: {overall_best['graph_config']}")
        print(f"Edge Features: {overall_best['use_edge_features']}")
        print(f"Transformer Layers: {overall_best['transformer_layers']}")
        print(f"Transformer Heads: {overall_best['transformer_heads']}")
        print(f"RMSE: {overall_best['metrics']['rmse']:.4f}")
        print(f"MAE: {overall_best['metrics']['mae']:.4f}")
    
    # Save results
    filename = f'transformer_experiments_{timestamp}.json'
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {filename}")
    
    # Create summary
    summary = {
        'timestamp': timestamp,
        'total_experiments': len(all_results),
        'configs_tested': configs_to_test,
        'transformer_layers_tested': transformer_layers,
        'transformer_heads_tested': transformer_heads,
        'best_per_config': {f"{k[0]}_{k[1]}": v for k, v in best_per_config.items()},
        'overall_best': overall_best if best_per_config else None
    }
    
    with open(f'transformer_summary_{timestamp}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to transformer_summary_{timestamp}.json")
    
    return summary
    
    
def optuna_optimize(config):
    """Fixed Optuna optimization that ensures config params are tested"""
    exp_folder = f"experimentations(restaurant)/{config.GRAPH_CONFIG}{'_transformer' if config.USE_TRANSFORMER else ''}{'_edge_features' if config.USE_EDGE_FEATURES else ''}"
    os.makedirs(exp_folder, exist_ok=True)
    # Prepare data once
    edge_list, n_users, n_items, builder = prepare_data(config)
    train_data, val_data, test_data = split_and_prepare_data(edge_list, builder, config)
    
    # Build config name for filename
    config_name = config.GRAPH_CONFIG
    if config.USE_TRANSFORMER:
        config_name += '_transformer'
    if config.USE_EDGE_FEATURES:
        config_name += '_edge_features'
    
    # Use timestamp for unique file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trials_file = f'{exp_folder}/{config_name}_trials_{timestamp}.json'
    
    def objective(trial):
        set_seed(14)
    
        # At the start of objective function
        complete_trials = trial.study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
        tried_params = {tuple(sorted(t.params.items())) for t in complete_trials}
        # Search spaces
        embedding_dim = trial.suggest_categorical("embedding_dim", [16, 32, 64, 128, 192, 256])
        num_layers = trial.suggest_categorical("num_layers", [1, 2, 3, 4])

        if config.USE_TRANSFORMER:
            transformer_layers = trial.suggest_categorical("transformer_layers", [1, 2, 3, 4])
            transformer_heads = trial.suggest_categorical("transformer_heads", [1, 2, 4, 8])
        else:
            transformer_layers = config.TRANSFORMER_LAYERS
            transformer_heads = config.TRANSFORMER_HEADS
    
        # After suggesting parameters
        current_params = tuple(sorted(trial.params.items()))
        if current_params in tried_params:
            raise optuna.TrialPruned()
        
        # Train with suggested params
        _, test_metrics = train_and_evaluate(
            config, train_data, val_data, test_data, n_users, n_items,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_epochs=config.NUM_EPOCHS,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads
        )
        
        # Save trial results with only required metrics
        trial_result = {
            'trial_number': trial.number,
            'timestamp': datetime.now().isoformat(),
            'params': trial.params,
            'rmse': test_metrics['rmse'],
            'mae': test_metrics['mae'],
            'mse': test_metrics['loss'],
            f'precision@{config.k}': test_metrics[f'precision@{config.k}'],
            f'recall@{config.k}': test_metrics[f'recall@{config.k}']
        }

        # Save to numbered trial file
        with open(trials_file, 'a') as f:
            f.write(json.dumps(trial_result) + '\n')

        return test_metrics['rmse'] 
   
        #sampler = optuna.samplers.TPESampler(seed=14, constant_liar=True)
        #study = optuna.create_study(direction="minimize", sampler=sampler)
        #study.optimize(objective, n_trials=20)

    sampler = optuna.samplers.TPESampler(seed=14, constant_liar=True)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # Keep trying until we get 20 successful (unique) trials
    n_successful_trials = 0
    while n_successful_trials < 20:
        try:
            study.optimize(objective, n_trials=1)
            n_successful_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        except:
            # In case all combinations have been tried
            print(f"Stopped at {n_successful_trials} trials - all unique combinations may have been explored")
            break

    # Get results
    print("\n=== Best Trial ===")
    print(f"Best RMSE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    
    # Train final model with best params
    best_params = study.best_params
    print("\n=== Training final model with best params ===")
    # Determine transformer parameters based on config
    if config.USE_TRANSFORMER:
        transformer_layers = best_params['transformer_layers']
        transformer_heads = best_params['transformer_heads']
    else:
        transformer_layers = config.TRANSFORMER_LAYERS
        transformer_heads = config.TRANSFORMER_HEADS
    final_model, final_metrics = train_and_evaluate(
        config, train_data, val_data, test_data, n_users, n_items,
        embedding_dim=best_params['embedding_dim'],
        num_layers=best_params['num_layers'],
        num_epochs=config.NUM_EPOCHS,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads
    )
    
    print("\n=== Final Results ===")
    print_metrics(final_metrics, k=config.k)
    
    # Use same config name and timestamp for best params file
    best_params_file = f'{exp_folder}/{config_name}_best_params_{timestamp}.json'
    
    # Save best results with only required metrics
    best_results = {
        'config': config_name,
        'best_params': best_params,
        'best_metrics': {
            'rmse': final_metrics['rmse'],
            'mae': final_metrics['mae'],
            'mse': final_metrics['loss'],
            f'precision@{config.k}': final_metrics[f'precision@{config.k}'],
            f'recall@{config.k}': final_metrics[f'recall@{config.k}']
        }
    }

    with open(best_params_file, 'w') as f:
        json.dump(best_results, f, indent=2)

    return best_params, final_metrics



def run_gamma_search(config):
    """
    Run gamma parameter search experiments.
    
    Args:
        config: Configuration object containing experiment parameters
    
    Returns:
        List of results containing gamma values and their corresponding metrics
    """

    

    gamma_values = [0.1,0.2, 0.3,0.4,0.5, 0.6,0.7, 0.8,0.9]
    
    results = []
    # Create config
    exp_config = copy.deepcopy(config)
    exp_config.GRAPH_CONFIG = 'ui_friend_trust'
    exp_config.EMBEDDING_DIM = 64
    exp_config.NUM_LAYERS = 1
    
    for gamma in gamma_values:
        set_seed(exp_config.SEED)  # Set random seed for reproducibility
        
        print(f"\n=== Testing gamma = {gamma} ===")
        

        exp_config.GAMMA = gamma

        
        # Run experiment
        model, metrics = main_global(exp_config)
        
        # Store results
        results.append({
            'gamma': gamma,
            'embedding_dim': exp_config.EMBEDDING_DIM,
            'num_layers': exp_config.NUM_LAYERS,
            'metrics': metrics
        })
    
    # Find best gamma
    best_result = min(results, key=lambda x: x['metrics']['rmse'])
    print(f"\n=== Best Gamma: {best_result['gamma']} with RMSE: {best_result['metrics']['rmse']:.4f} ===")
    
    # Save results
    os.makedirs('grid_social_shopping/ui_friend_trust', exist_ok=True)
    with open('grid_social_shopping/ui_friend_trust/gamma_grid_search.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results






def exp_GCN_up_s(config):
    
    # Define graph configurations to test
    graph_configs = ['ui','ui_trust_item']  # Order matters for consistency
    seed=config.SEED

    # Grid search parameters
    embedding_dims = [16, 32, 64, 128, 192, 256]  # Embedding dimensions to test
    num_layers_list = [1,2,3,4]
    alpha_values = config.ALPHA  # Use predefined alpha values from config
    
    # Store results for all configurations
    all_results = {}
    
    # Add timestamp to avoid overwriting previous results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Iterate over each graph configuration
    for graph_config in graph_configs:

        print(f"\n{'='*50}")
        print(f"Testing Graph Configuration: {graph_config}")
        print(f"{'='*50}\n")
        
        results = []
        
        # Test each combination for current graph config
        for emb_dim in embedding_dims:
            for n_layers in num_layers_list:
  
                    print(f"\n=== Testing {graph_config}: emb_dim={emb_dim}, layers={n_layers} ===")
                    
                    # Create experiment-specific config
                    trial_config = copy.deepcopy(config)
                    trial_config.GRAPH_CONFIG = graph_config
                    trial_config.EMBEDDING_DIM = emb_dim
                    trial_config.NUM_LAYERS = n_layers
                    alpha=config.ALPHA
                    # Ensure transformer is off for base GCN experiments
                    trial_config.USE_TRANSFORMER = False
                    trial_config.USE_EDGE_FEATURES = True
                    
                    try:
                        # Prepare data for this specific alpha configuration
                        ui_edges, ui_weights, social_edges_list, n_users, n_items, builder = prepare_data(trial_config)
                        train_data, val_data, test_data = split_and_prepare_data(
                            ui_edges, ui_weights, social_edges_list, builder, trial_config, n_users, n_items
                        )
                        
                        # Train and evaluate
                        model, test_metrics = train_and_evaluate(
                            trial_config, train_data, val_data, test_data, n_users, n_items
                        )
                        
                        print_metrics(test_metrics, k=trial_config.k)
                        print(f"RMSE: {test_metrics['rmse']:.4f}, MAE: {test_metrics['mae']:.4f}, Alpha: {alpha}")
                        
                        # Store additional info for analysis
                        results.append({
                            'graph_config': graph_config,
                            'emb_dim': emb_dim,
                            'num_layers': n_layers,
                            'alpha': alpha,  # Store alpha value
                            'metrics': test_metrics,
                            'training_time': test_metrics.get('training_time', None)
                        })
                        
                    except Exception as e:
                        print(f"Error in configuration: {e}")
                        results.append({
                            'graph_config': graph_config,
                            'emb_dim': emb_dim,
                            'num_layers': n_layers,
                            'alpha': alpha,  # Store alpha even for errors
                            'metrics': {'rmse': float('inf'), 'mae': float('inf')},
                            'error': str(e)
                        })
        
        # Store results for this graph config
        all_results[graph_config] = results
        
        # Find best for this graph config (excluding failed runs)
        valid_results = [r for r in results if 'error' not in r]
        if valid_results:
            best_result = min(valid_results, key=lambda x: x['metrics']['rmse'])
            print(f"\n=== Best for {graph_config}: emb_dim={best_result['emb_dim']}, "
                  f"layers={best_result['num_layers']}, alpha={best_result['alpha']}, "
                  f"RMSE={best_result['metrics']['rmse']:.4f} ===")
        
        # Save results with timestamp
        filename = f'grid_search_results_{graph_config}_sim_{config.MIN_SIMILARITY}_jacc_{config.MIN_JACCARD}_{timestamp}.json'
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filename}")
    
    # Find overall best configuration
    print(f"\n{'='*50}")
    print("OVERALL BEST CONFIGURATION")
    print(f"{'='*50}")
    
    overall_best = None
    overall_best_rmse = float('inf')
    
    for graph_config, results in all_results.items():
        valid_results = [r for r in results if 'error' not in r]
        if valid_results:
            best_in_config = min(valid_results, key=lambda x: x['metrics']['rmse'])
            if best_in_config['metrics']['rmse'] < overall_best_rmse:
                overall_best_rmse = best_in_config['metrics']['rmse']
                overall_best = best_in_config
    
    if overall_best:
        print(f"Best overall: Graph={overall_best['graph_config']}, "
              f"emb_dim={overall_best['emb_dim']}, layers={overall_best['num_layers']}, "
              f"alpha={overall_best['alpha']}, RMSE={overall_best['metrics']['rmse']:.4f}")
    
    # Save all results with timestamp
    with open(f'all_grid_search_results_{timestamp}.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create analysis-friendly summary
    summary = {
        'timestamp': timestamp,
        'total_experiments': sum(len(results) for results in all_results.values()),
        'graph_configs_tested': list(all_results.keys()),
        'embedding_dims_tested': embedding_dims,
        'num_layers_tested': num_layers_list,
        'alpha_values_tested': alpha_values,  # Add alpha values to summary
        'best_per_graph': {},
        'overall_best': overall_best
    }
    
    for graph_config, results in all_results.items():
        valid_results = [r for r in results if 'error' not in r]
        if valid_results:
            best = min(valid_results, key=lambda x: x['metrics']['rmse'])
            summary['best_per_graph'][graph_config] = {
                'emb_dim': best['emb_dim'],
                'num_layers': best['num_layers'],
                'alpha': best['alpha'],  # Include alpha in summary
                'rmse': best['metrics']['rmse'],
                'mae': best['metrics']['mae'],
                'all_metrics': best['metrics']
            }
    
    with open(f'grid_search_summary_{timestamp}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nAll results saved!")
    print(f"- Individual graph results: grid_search_results_[graph_config]_{timestamp}.json")
    print(f"- All results: all_grid_search_results_{timestamp}.json")
    print(f"- Summary: grid_search_summary_{timestamp}.json")
    print(f"min similairy for trust edges is set to {config.MIN_SIMILARITY} and min items for trust edges is set to {config.MIN_COMMON_ITEMS}")
    
    return summary









def run_gamma_search(config):
    """
    Run gamma parameter search experiments.
    
    Args:
        config: Configuration object containing experiment parameters
    
    Returns:
        List of results containing gamma values and their corresponding metrics
    """

    

    gamma_values = [0.1,0.2, 0.3,0.4,0.5, 0.6,0.7, 0.8,0.9]
    
    results = []
    # Create config
    exp_config = copy.deepcopy(config)
    exp_config.GRAPH_CONFIG = 'ui_friend_trust'
    exp_config.EMBEDDING_DIM =16
    exp_config.NUM_LAYERS = 1
    
    for gamma in gamma_values:
        set_seed(exp_config.SEED)  # Set random seed for reproducibility
        
        print(f"\n=== Testing gamma = {gamma} ===")
        

        exp_config.GAMMA = gamma

        
        # Run experiment
        model, metrics = main_global(exp_config)
        
        # Store results
        results.append({
            'gamma': gamma,
            'embedding_dim': exp_config.EMBEDDING_DIM,
            'num_layers': exp_config.NUM_LAYERS,
            'metrics': metrics
        })
    
    # Find best gamma
    best_result = min(results, key=lambda x: x['metrics']['rmse'])
    print(f"\n=== Best Gamma: {best_result['gamma']} with RMSE: {best_result['metrics']['rmse']:.4f} ===")
    
    # Save results
    os.makedirs('grid_social_shopping/ui_friend_trust', exist_ok=True)
    with open('grid_social_shopping/ui_friend_trust/gamma_grid_search.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def exp_transformer_up(config):
    
    # Define specific configurations to test
    configs_to_test = [
        #'GRAPH_CONFIG': 'ui', 'USE_TRANSFORMER': True, 'USE_EDGE_FEATURES': False,
        # 'EMBEDDING_DIM': 64, 'NUM_LAYERS': 1},
        {'GRAPH_CONFIG': 'ui', 'USE_TRANSFORMER': True, 'USE_EDGE_FEATURES': True,
         'EMBEDDING_DIM': 64, 'NUM_LAYERS': 1},
        #{'GRAPH_CONFIG': 'ui_friend_trust', 'USE_TRANSFORMER': True, 'USE_EDGE_FEATURES': False,
        # 'EMBEDDING_DIM': 64, 'NUM_LAYERS': 1, 'GAMMA': 0.9},
        {'GRAPH_CONFIG': 'ui_friend_trust', 'USE_TRANSFORMER': True, 'USE_EDGE_FEATURES': True,
         'EMBEDDING_DIM': 64, 'NUM_LAYERS': 1, 'GAMMA': 0.9}
    ]
    
    seed = config.SEED
    
    # Grid search parameters for transformer
    transformer_layers_list = [1, 2, 3, 4]
    num_heads_list = [1,2, 4, 8]
    
    # Store results for all configurations
    all_results = {}
    
    # Add timestamp to avoid overwriting previous results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Iterate over each base configuration
    for base_config_idx, base_config in enumerate(configs_to_test):
        set_seed(seed)
        config_name = f"{base_config['GRAPH_CONFIG']}_edge_{base_config['USE_EDGE_FEATURES']}"
        
        print(f"\n{'='*60}")
        print(f"Testing Base Configuration {base_config_idx + 1}/{len(configs_to_test)}: {config_name}")
        print(f"Graph: {base_config['GRAPH_CONFIG']}, Edge Features: {base_config['USE_EDGE_FEATURES']}")
        print(f"Fixed: emb_dim={base_config['EMBEDDING_DIM']}, layers={base_config['NUM_LAYERS']}")
        if 'GAMMA' in base_config:
            print(f"Gamma: {base_config['GAMMA']}")
        print(f"{'='*60}\n")
        
        # Create a copy of config and apply base configuration
        exp_config = copy.deepcopy(config)
        for key, value in base_config.items():
            setattr(exp_config, key, value)
        
        # Prepare data for this configuration
        ui_edges, ui_weights, social_edges_list, n_users, n_items, builder, business_df = prepare_data(exp_config)
        train_data, val_data, test_data = split_and_prepare_data(
            ui_edges, ui_weights, social_edges_list, builder, exp_config, n_users, n_items, business_df
        )
        
        results = []
        
        # Test each transformer configuration
        for transformer_layers in transformer_layers_list:
            for num_heads in num_heads_list:
                
                print(f"\n=== Testing {config_name}: transformer_layers={transformer_layers}, heads={num_heads} ===")
            
                # Create experiment-specific config
                trial_config = copy.deepcopy(exp_config)
                trial_config.TRANSFORMER_LAYERS = transformer_layers
                trial_config.NUM_HEADS = num_heads
                
                try:
                    # Train and evaluate
                    model, test_metrics = train_and_evaluate(
                        trial_config, train_data, val_data, test_data, n_users, n_items
                    )
                    
                    print_metrics(test_metrics, k=trial_config.k)
                    print(f"RMSE: {test_metrics['rmse']:.4f}, MAE: {test_metrics['mae']:.4f}")
                    
                    # Store all configuration details for analysis
                    result_entry = {
                        'base_config_idx': base_config_idx,
                        'config_name': config_name,
                        'transformer_layers': transformer_layers,
                        'num_heads': num_heads,
                        'metrics': test_metrics,
                        'training_time': test_metrics.get('training_time', None)
                    }
                    
                    # Add all base config parameters to result
                    for key, value in base_config.items():
                        result_entry[key.lower()] = value
                    
                    results.append(result_entry)
                    
                except Exception as e:
                    print(f"Error in configuration: {e}")
                    error_entry = {
                        'base_config_idx': base_config_idx,
                        'config_name': config_name,
                        'transformer_layers': transformer_layers,
                        'num_heads': num_heads,
                        'metrics': {'rmse': float('inf'), 'mae': float('inf')},
                        'error': str(e)
                    }
                    
                    # Add base config parameters
                    for key, value in base_config.items():
                        error_entry[key.lower()] = value
                    
                    results.append(error_entry)
        
        # Store results for this base config
        all_results[config_name] = results
        
        # Find best for this base config (excluding failed runs)
        valid_results = [r for r in results if 'error' not in r]
        if valid_results:
            best_result = min(valid_results, key=lambda x: x['metrics']['rmse'])
            print(f"\n=== Best for {config_name}: transformer_layers={best_result['transformer_layers']}, "
                  f"heads={best_result['num_heads']}, RMSE={best_result['metrics']['rmse']:.4f} ===")
        
        # Save results with timestamp
        filename = f'transformer_config_{base_config_idx}_{config_name}_{timestamp}.json'
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filename}")
    
    # Find overall best configuration
    print(f"\n{'='*60}")
    print("OVERALL BEST TRANSFORMER CONFIGURATION")
    print(f"{'='*60}")
    
    overall_best = None
    overall_best_rmse = float('inf')
    
    for config_name, results in all_results.items():
        valid_results = [r for r in results if 'error' not in r]
        if valid_results:
            best_in_config = min(valid_results, key=lambda x: x['metrics']['rmse'])
            if best_in_config['metrics']['rmse'] < overall_best_rmse:
                overall_best_rmse = best_in_config['metrics']['rmse']
                overall_best = best_in_config
    
    if overall_best:
        print(f"Best overall configuration:")
        print(f"  - Base config: {overall_best['config_name']}")
        print(f"  - Graph: {overall_best['graph_config']}")
        print(f"  - Edge features: {overall_best['use_edge_features']}")
        print(f"  - Transformer layers: {overall_best['transformer_layers']}")
        print(f"  - Attention heads: {overall_best['num_heads']}")
        print(f"  - RMSE: {overall_best['metrics']['rmse']:.4f}")
        if 'gamma' in overall_best:
            print(f"  - Gamma: {overall_best['gamma']}")
    
    # Save all results with timestamp
    with open(f'all_transformer_configs_results_{timestamp}.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create analysis-friendly summary
    summary = {
        'timestamp': timestamp,
        'experiment_type': 'transformer_configs_grid_search',
        'base_configs_tested': len(configs_to_test),
        'configs_to_test': configs_to_test,
        'transformer_layers_tested': transformer_layers_list,
        'num_heads_tested': num_heads_list,
        'total_experiments': sum(len(results) for results in all_results.values()),
        'best_per_config': {},
        'overall_best': overall_best
    }
    
    for config_name, results in all_results.items():
        valid_results = [r for r in results if 'error' not in r]
        if valid_results:
            best = min(valid_results, key=lambda x: x['metrics']['rmse'])
            summary['best_per_config'][config_name] = {
                'transformer_layers': best['transformer_layers'],
                'num_heads': best['num_heads'],
                'rmse': best['metrics']['rmse'],
                'mae': best['metrics']['mae'],
                'all_metrics': best['metrics'],
                'base_config': {k: v for k, v in best.items() if k not in ['metrics', 'transformer_layers', 'num_heads', 'config_name', 'base_config_idx', 'training_time']}
            }
    
    with open(f'transformer_configs_summary_{timestamp}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nAll transformer configuration results saved!")
    print(f"- Individual config results: transformer_config_[idx]_[name]_{timestamp}.json")
    print(f"- All results: all_transformer_configs_results_{timestamp}.json")
    print(f"- Summary: transformer_configs_summary_{timestamp}.json")
    print(f"Total configurations tested: {len(configs_to_test)} base configs × {len(transformer_layers_list)} layers × {len(num_heads_list)} heads = {len(configs_to_test) * len(transformer_layers_list) * len(num_heads_list)} experiments")
    
    return summary  # Return summary for further processing