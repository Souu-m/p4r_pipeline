"""Training and evaluation utilities"""

import copy
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, mean_absolute_error,
                             mean_squared_error, precision_score, recall_score)


def evaluate_model(model, data,train_data,n_users ,k=20, threshold=3.5):
    """
    Original evaluation: predicts ratings for specific user-item pairs in test set
    """
    model.eval()
    
    device = next(model.parameters()).device
    data = data.to(device)
    
    user_indices = data.edge_index[0]
    item_indices = data.edge_index[1]
    #MASK U-U and Biderectionnal ARCS
    ui_mask = (user_indices < n_users) & (item_indices >= n_users)
    ui_user_indices = user_indices[ui_mask]
    ui_item_indices = item_indices[ui_mask]
    
    true_ratings = data.y[ui_mask].cpu().numpy()

    with torch.no_grad():
        predicted_ratings = model.predict_base(ui_user_indices, ui_item_indices, train_data).squeeze().cpu().numpy()


    rmse = np.sqrt(mean_squared_error(true_ratings, predicted_ratings))
    mae = mean_absolute_error(true_ratings, predicted_ratings)

    
    recall_at_k, precision_at_k=get_recall_at_k(
        input_edge_index=torch.stack([ui_user_indices, ui_item_indices]),
        input_edge_values=true_ratings,
        pred_ratings=predicted_ratings,
        k=k,
        threshold=threshold
        )
    
    #recall_at_k, precision_at_k=calculate_metrics_at_k(ui_user_indices, ui_item_indices, true_ratings, predicted_ratings, k=k, threshold=threshold)
    mse = mean_squared_error(true_ratings, predicted_ratings)

    # --- Add F1 score ---
    if (precision_at_k + recall_at_k) > 0:
        f1_at_k = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)
    else:
        f1_at_k = 0.0
    metrics = {
        'rmse': rmse,
        'mae': mae,
        f'precision@{k}': precision_at_k,
        f'recall@{k}': recall_at_k,
        f'f1@{k}': f1_at_k,
        'loss': mse
    }
    
    return metrics

def print_metrics(metrics, k=20):
    """Print evaluation metrics with an optional prefix."""
    print(f" RMSE: {metrics['rmse']:.4f}")
    print(f" MAE: {metrics['mae']:.4f}")
    print(f" MSE: {metrics['loss']:.4f}")
    print(f" Precision@{k}: {metrics[f'precision@{k}']:.4f}")
    print(f" Recall@{k}: {metrics[f'recall@{k}']:.4f}")  
    print(f" f1@{k}: {metrics[f'f1@{k}']:.4f}")  
    


def get_recall_at_k(input_edge_index, 
                     input_edge_values, # the true label of actual ratings for each user/item interaction
                     pred_ratings, # the list of predicted ratings
                     k=20, 
                     threshold=3.5):
    with torch.no_grad():
        user_item_rating_list = defaultdict(list)

        for i in range(len(input_edge_index[0])):
            src = input_edge_index[0][i].item()
            dest = input_edge_index[1][i].item()
            true_rating = input_edge_values[i].item()
            pred_rating = pred_ratings[i].item()

            user_item_rating_list[src].append((pred_rating, true_rating))

        recalls = dict()
        precisions = dict()

        for user_id, user_ratings in user_item_rating_list.items():
            user_ratings.sort(key=lambda x: x[0], reverse=True)

            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
            
            n_rel_and_rec_k = sum(
                ((true_r >= threshold) and (est >= threshold))
                for (est, true_r) in user_ratings[:k]
            )

            precisions[user_id] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
            recalls[user_id] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

        overall_recall = sum(rec for rec in recalls.values()) / len(recalls)
        overall_precision = sum(prec for prec in precisions.values()) / len(precisions)

        return overall_recall, overall_precision
    
    
