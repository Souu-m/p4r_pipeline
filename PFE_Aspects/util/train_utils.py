import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from util.evaluation_utils import evaluate_model


def train_model(model, optimizer, train_loader, val_data, criterion, config, n_users,
                num_epochs, patience, mode, aspect_name=None, aspect_data_dict_train=None,aspect_data_dict_val=None):
    """
    Train the GCN model with early stopping - now supports both global and aspect training
    
    Args:
        mode: 'global' or 'aspects'
        mode 'global': Train the model on the global dataset (all interactions that are not aspect-specific)
        aspect_name: Name of the aspect being trained 
        aspect_data_dict: Dict containing aspect data 
    """
    device = next(model.parameters()).device
    val_data = val_data.to(device)
    
    best_val_rmse = float('inf')
    best_model_state = None
    no_improve_count = 0
    train_losses = []
    val_losses = []
    min_delta = 1e-4
    
    # Print training mode
    if mode == 'aspects' :
        print(f"\nTraining aspect: {aspect_name}")
    else:
        print("\nTraining global model")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Training phase
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            user_indices = data.edge_index[0]
            item_indices = data.edge_index[1]
            #MASK U-U and Biderectionnal ARCS ( This option is for when we use social information only & it does not affect the training) 
            #[The Idea of the Mask is to use social information only for propagation in order to enrich the user embeddings only then the preidction is done on the user-item interactions only not on the user-user interactions]
            ui_mask = (user_indices < n_users) & (item_indices >= n_users)
            ui_user_indices = user_indices[ui_mask]
            ui_item_indices = item_indices[ui_mask]
            
            # Handle prediction based on mode
            predictions = model.predict_base(
                    ui_user_indices, ui_item_indices,
                    data,
                    mode='global'
                ).squeeze()
            
            loss = criterion(predictions, data.y[ui_mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)
        
        
        val_metrics = evaluate_model(
                model, val_data, data,n_users,
                k=config.k, threshold=config.Threshold
            )
            
        val_loss = val_metrics['loss']
        val_rmse = val_metrics['rmse']
        val_losses.append(val_loss)
        
        # Check if validation performance improved
        # Early stopping based on RMSE (as NEUMF paper)
        # Paper: "stopped training if the RMSE on validation set increased for 5 successive epochs"
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_state = copy.deepcopy(model.state_dict())
            no_improve_count = 0
            '''
            print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f} (best)")
            '''
        else:
            no_improve_count += 1
            '''
            print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f} "
                f"(no improvement: {no_improve_count}/{patience})")
            '''
            if no_improve_count >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Best validation rmse: {best_val_rmse:.4f}")
    
    return model