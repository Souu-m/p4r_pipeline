import torch

# Load the dictionary
item_embeddings = torch.load("item_embeddings.pt",weights_only=True)

# Print type and number of entries
print("Type:", type(item_embeddings))
print("Number of items:", len(item_embeddings))

# Print first 10 keys
keys = list(item_embeddings.keys())
print("First 10 keys:", keys[:10])

# Check if the first key is 0
print("Starts from 0?", keys[0] == 0)

# Optionally print shape of first embedding
first_key = keys[0]
print("First key:", first_key)
print("Embedding shape:", item_embeddings[first_key].shape)
