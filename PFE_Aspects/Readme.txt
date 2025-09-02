Hey! Here's a guide to help you navigate through my recommendation system project.
1. Run `python main.py` to see current results



### Main Files to Focus On

1. **`main.py`** - Entry point
   - `main_global()` - runs baseline (global graph only) 
   - `train_and_evaluate_aspects_based()` - runs unified model (global + aspects)
   - Just run this file to see current results in Terminal

2. **`models.py`** - The SENGR_GCN model 
   - it contain mainly GCN model for propagation and SENGR_GCN class that calls GCN and (optionally transformer for propagation) and handle prediction module
   - Look at `predict()` method - this is where concatenation of aspects + base model happens
   - `forward()` - processes global graph ( forward handle the propagation )
   - `forward_aspect()` - processes each aspect graph

3. **`config.py`** - Settings
   - Make sure `GRAPH_CONFIG = 'ui'` (no social edges for the moment)  
   - You can test with transformer True to see the difference and degragdation

4. **`aspects_graphs_utils.py`** - How aspect graphs are built
   - Creates subgraphs from the main graph split
   - Only includes edges where aspect was mentioned

### The Data Flow

```
Dataset/preprocessed/
├── users.csv (3,720 users)
├── businesses.csv (11,301 items)  
├── reviews.csv (71,358 ratings)
└── reviews_with_aspects.csv (same reviews + aspect lists)
```


### Running the Code

```bash
# This runs both experiments (base model (global graph) + second mode with aspects based which mean running base model with aspect based and concatenate their embeddings )
python main.py

# You'll see:
# 1. Global only results (baseline)
# 2. Global + Aspects results (should be better but currently isn't)
```

### The Problem I Need Help With

When you run it, you'll notice with transformer:
- Global model: RMSE ≈ 0.6 ✓
- Global + Aspects: RMSE ≈ worse... 

The issue is in how we combine embeddings. Look at `models.py` in the `predict()` method:

```python
# Around line 200-250
if mode in ['aspects', 'all']:
    for aspect, data in aspect_data_dict.items():
        # Gets embeddings for ALL users/items
        aspect_embeddings = self.forward_aspect(aspect, data.edge_index, data.edge_weight)
        user_emb = aspect_embeddings[user_indices]  # But some users never mentioned this aspect!
        item_emb = aspect_embeddings[item_indices]  # Their embeddings are random!
```

### What I Think We Should Try

1. **Quick Fix - Filtering**:
   In `predict()`, only use aspect embeddings if user/item actually mentioned that aspect. Maybe maintain a mapping of which users mentioned which aspects.

2. **Better Architecture**:
   Instead of concatenating everything, maybe use attention or weighted sum based on whether the aspect was mentioned.



### Where to Start


The key insight: We're concatenating embeddings from users who never mentioned certain aspects (so their aspect embeddings are untrained/random), which adds noise.
