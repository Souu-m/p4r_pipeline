import torch
import torch.nn as nn


def projection(file_path, output_dim=64, use_relu=False, save_path=None):
    semantic_dict = torch.load(file_path)
    
    # Get input dimension from first embedding
    input_dim = next(iter(semantic_dict.values())).shape[-1]
    
    # Create and apply projection
    proj = nn.Linear(input_dim, output_dim)
    
    projected = {}
    with torch.no_grad():
        for item_id, emb in semantic_dict.items():
            projected_emb = proj(emb)
            if use_relu:
                projected_emb = torch.relu(projected_emb)
            projected[item_id] = projected_emb
    
    # save the projected embeddings
    if save_path is not None:
        torch.save(projected, save_path)
        print(f"Projected embeddings saved to {save_path}")
    
    return projected



if __name__ == "__main__":
    # Simple linear projection to 64D without ReLU
    projected = projection(
        'item_embeddings.pt',
        output_dim=64,
        save_path='projected_item_embeddings_64.pt'
    )

    # With ReLU activation
    projected_relu = projection('item_embeddings.pt', output_dim=64, use_relu=True, save_path='projected_item_embeddings_64_relu.pt')

