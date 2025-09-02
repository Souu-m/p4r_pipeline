from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import torch
import pandas as pd
import numpy as np
from collections import defaultdict


import torch
import pandas as pd
import numpy as np
from collections import defaultdict
from util.data_utils import standardize_aspects

class HeterogeneousGraphBuilder:
    """
    Modular edge construction for flexible experimentation with different edge types
    """
    
    def __init__(self, n_users, n_items):
        self.n_users = n_users
        self.n_items = n_items
        self.edge_builders = {}
        
    def build_user_item_edges(self, review_df, min_rating=1, bidirectional=False):
       """
       Build user-item interaction edges (base edges)
       
       Args:
           review_df: DataFrame with user_id_encoded, item_id_encoded, rating
           min_rating: Minimum rating to create an edge
           bidirectional: If True, creates edges in both directions (user->item and item->user)
           
       Returns:
           edge_index, edge_weight
       """
       print(f"Building User-Item edges (bidirectional={bidirectional})...")
       
       # Filter by minimum rating
       interactions = review_df[review_df['rating'] >= min_rating]
       
       # Create base edges (user -> item)
       user_ids = interactions['user_id'].values
       item_ids = interactions['item_id'].values
       ratings = interactions['rating'].values
       
       if bidirectional:
           # Create bidirectional edges: user->item AND item->user
           edge_index = torch.stack([
               torch.LongTensor(np.concatenate([user_ids, item_ids])),
               torch.LongTensor(np.concatenate([item_ids, user_ids]))
           ])
           edge_weight = torch.FloatTensor(np.concatenate([ratings, ratings]))
       else:
           # Unidirectional edges: user->item only
           edge_index = torch.stack([
               torch.LongTensor(user_ids),
               torch.LongTensor(item_ids)
           ])
           edge_weight = torch.FloatTensor(ratings)
       
       print(f"Created {edge_index.shape[1]} edges")
       return edge_index, edge_weight
    
    
    def build_user_user_friendship_edges(self, user_df, user_encoder, min_jaccard=0.0):
        """
        Build user-user edges for ALL direct friendships
        Weight = Jaccard coefficient (computed on ALL friends, including non-encoded ones)
        Optimized to O(n*m) where n = users, m = average friends per user
        """
        print("Building User-User friendship edges based on Jaccard similarity...")
        print(f'Note: ALL direct friendships will be included, weight = Jaccard')
        
        # Create bidirectional mappings
        encoded_to_user_id = dict(enumerate(user_encoder.classes_))
        user_id_to_encoded = {v: k for k, v in encoded_to_user_id.items()}
        
        # Build friend sets using STRING IDs for complete Jaccard calculation
        user_friend_sets = {}  # {encoded_idx: set of string friend IDs}
        users_with_friends = user_df[user_df['friends'].notna() & (user_df['friends'] != 'None')]
        
        for _, user in users_with_friends.iterrows():
            encoded_idx = user['user_id']  # Already encoded
            friends_str = str(user['friends']).split(',')
            
            # Keep ALL friends as strings for Jaccard (including non-encoded ones)
            friend_set = {f.strip() for f in friends_str if f.strip()}
            
            if friend_set:
                user_friend_sets[encoded_idx] = friend_set
        
        # Build edges - iterate only through actual friendships
        sources = []
        targets = []
        weights = []
        processed_pairs = set()
        
        # For each user, check ONLY their direct friends (not all users)
        for user_a_encoded, friends_a_str in user_friend_sets.items():
            user_a_str = encoded_to_user_id[user_a_encoded]
            
            # Iterate only through user_a's friends
            for friend_b_str in friends_a_str:
                # Check if this friend is in our encoded users (in the graph)
                if friend_b_str not in user_id_to_encoded:
                    continue  # Friend not in graph, skip edge creation
                
                user_b_encoded = user_id_to_encoded[friend_b_str]
                
                # Avoid processing the same pair twice
                if user_a_encoded >= user_b_encoded:
                    continue  # Will be processed when user_b is user_a
                
                # Mark as processed
                pair = (user_a_encoded, user_b_encoded)
                if pair in processed_pairs:
                    continue
                processed_pairs.add(pair)
                
                # Calculate Jaccard using ALL friends (including non-encoded ones)
                friends_b_str = user_friend_sets.get(user_b_encoded, set())
                
                if friends_b_str:
                    intersection = len(friends_a_str & friends_b_str)
                    union = len(friends_a_str | friends_b_str)
                    jaccard = intersection / union if union > 0 else 0.0
                else:
                    # user_b has no friends list, but is friend of user_a
                    # Jaccard = 0 / len(friends_a) = 0
                    jaccard = 0.0
                
                # Create edges only if Jaccard > threshold
                if jaccard > min_jaccard:
                    # Create bidirectional edges
                    sources.extend([user_a_encoded, user_b_encoded])
                    targets.extend([user_b_encoded, user_a_encoded])
                    weights.extend([jaccard, jaccard])
        
        if sources:
            edge_index = torch.stack([
                torch.LongTensor(sources),
                torch.LongTensor(targets)
            ])
            edge_weight = torch.FloatTensor(weights)
            
            # Statistics (use every other weight since edges are bidirectional)
            unique_weights = weights[::2]
            avg_jaccard = sum(unique_weights) / len(unique_weights)
            min_weight = min(unique_weights)
            max_weight = max(unique_weights)
            
            print(f"Created {edge_index.shape[1]} directed edges ({len(processed_pairs)} unique friendships)")
            print(f"Jaccard weights: min={min_weight:.3f}, avg={avg_jaccard:.3f}, max={max_weight:.3f}")
            
            return edge_index, edge_weight
        else:
            print("No friendship connections found above threshold")
            return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float)


    def build_user_user_trust_edges(self, ui_edge_index, ui_edge_weights, min_common_items=10, min_similarity=0.3):
        """
        Build user-user trust edges based on cosine similarity using existing UI edges
        Optimized version using sparse matrices.
        """
        print("Building User-User trust edges based on similarity...")
        print(f"Using {min_common_items} common items and {min_similarity} minimum similarity")
        
        # Get users and items
        users = ui_edge_index[0].numpy()
        items = ui_edge_index[1].numpy()
        ratings = ui_edge_weights.numpy()
        
        # Get dimensions
        max_user = users.max() + 1
        max_item = items.max() + 1
        
        # Build user-item rating matrix (sparse)
        from scipy.sparse import csr_matrix
        user_item_matrix = csr_matrix((ratings, (users, items)), shape=(max_user, max_item))
        
        # Filter users with enough items (early pruning)
        user_item_counts = np.array(user_item_matrix.getnnz(axis=1)).flatten()
        valid_users = np.where(user_item_counts >= min_common_items)[0]
        
        if len(valid_users) < 2:
            print("Not enough users with minimum items")
            return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float)
        
        # Filter matrix to valid users only
        filtered_matrix = user_item_matrix[valid_users]
        
        # Compute cosine similarity using matrix operations
        # Normalize rows (users) to unit vectors
        from sklearn.preprocessing import normalize
        normalized_matrix = normalize(filtered_matrix, norm='l2', axis=1)
        
        # Compute similarity matrix (only upper triangle to avoid duplicates)
        similarity_matrix = normalized_matrix @ normalized_matrix.T
        
        # Convert to coordinate format for easier processing
        similarity_coo = similarity_matrix.tocoo()
        
        sources = []
        targets = []
        weights = []
        
        # Process similarity results
        for i, j, sim in zip(similarity_coo.row, similarity_coo.col, similarity_coo.data):
            if i < j and sim >= min_similarity:  # Only upper triangle, avoid self-loops
                # Check common items count (more precise check)
                user_i_items = set(filtered_matrix[i].nonzero()[1])
                user_j_items = set(filtered_matrix[j].nonzero()[1])
                common_count = len(user_i_items & user_j_items)
                
                if common_count >= min_common_items:
                    # Map back to original user IDs
                    orig_user_i = valid_users[i]
                    orig_user_j = valid_users[j]
                    
                    # Add bidirectional edges
                    sources.extend([orig_user_i, orig_user_j])
                    targets.extend([orig_user_j, orig_user_i])
                    weights.extend([sim, sim])
        
        if sources:
            edge_index = torch.stack([
                torch.LongTensor(sources),
                torch.LongTensor(targets)
            ])
            edge_weight = torch.FloatTensor(weights)
            print(f"Created {len(sources)//2} trust connections")
            return edge_index, edge_weight
        else:
            print("No trust connections created")
            return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float)
        
    def build_user_user_trust_edges_v1(self, ui_edge_index, ui_edge_weights, min_common_items=10, min_similarity=0.3):
        """
        Build user-user trust edges based on cosine similarity using existing UI edges

        Args:
            ui_edge_index: Tensor of shape [2, num_edges] with user-item interactions
            ui_edge_weights: Tensor of edge weights (ratings)
            min_common_items: Minimum common items rated
            min_similarity: Minimum cosine similarity

        Returns:
            edge_index, edge_weight
        """
        print("Building User-User trust edges based on similarity...")
        print(f"Using {min_common_items} common items and {min_similarity} minimum similarity")
        # Extract users and items from edge index
        users = ui_edge_index[0]
        items = ui_edge_index[1]
        ratings = ui_edge_weights

        # Build user->items mapping efficiently
        user_to_items = {}
        for idx in range(ui_edge_index.shape[1]):
            u = users[idx].item()
            i = items[idx].item()
            r = ratings[idx].item()

            if u not in user_to_items:
                user_to_items[u] = {}
            user_to_items[u][i] = r

        # Get unique users
        unique_users = sorted(user_to_items.keys())
        n_users = len(unique_users)

        weights = []
        sources = []
        targets = []    


        # Compute pairwise similarity
        for i, u in enumerate(unique_users):

            for j in range(i + 1, n_users):
                u_prime = unique_users[j]

                # Get items and ratings for both users
                items_u = set(user_to_items[u].keys())
                items_u_prime = set(user_to_items[u_prime].keys())

                # Find common items
                common_items = items_u & items_u_prime

                if len(common_items) >= min_common_items:
                    # Calculate cosine similarity
                    dot_product = 0
                    norm_u_sq = 0
                    norm_u_prime_sq = 0

                    for item in common_items:
                        r_u = user_to_items[u][item]
                        r_u_prime = user_to_items[u_prime][item]

                        dot_product += r_u * r_u_prime
                        norm_u_sq += r_u * r_u
                        norm_u_prime_sq += r_u_prime * r_u_prime

                    # Compute similarity
                    if norm_u_sq > 0 and norm_u_prime_sq > 0:
                        similarity = dot_product / (np.sqrt(norm_u_sq) * np.sqrt(norm_u_prime_sq))

                        if similarity >= min_similarity:
                            # Add bidirectional edges
                            sources.append(u)  # user_a → user_b
                            targets.append(u_prime) 
                            weights.append(similarity)

                            sources.append(u_prime)  # u_prime → u (reverse direction)
                            targets.append(u)
                            weights.append(similarity)
    

        if sources and targets:
            edge_index = torch.stack([
                torch.LongTensor(sources),  # Row 0: Sources
                torch.LongTensor(targets)   # Row 1: Targets
            ])
            edge_weight = torch.FloatTensor(weights)
            print(f"Created {edge_index.shape[1]//2} trust connections")
        else:
            #edge_index = torch.empty((2, 0), dtype=torch.long)
            #edge_weight = torch.empty(0, dtype=torch.float)
            print("No trust connections created")

        return edge_index, edge_weight


    def build_item_item_similarity_edges(self, business_df, min_similarity=0.5, top_k=5):
        """
        Build item-item edges based on attribute similarity (city + categories)
        Using the same pattern as user-user edges
        
        Args:
            business_df: DataFrame with item info including city and categories
            min_similarity: Minimum cosine similarity threshold
            top_k: Keep only top-k most similar items for each item
        
        Returns:
            edge_index, edge_weight
        """
        print("Building Item-Item similarity edges based on attributes...")
        print(f"Using min_similarity={min_similarity}, top_k={top_k}")
        
        # Prepare item features
        item_features = self._prepare_item_features(business_df)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(item_features)
        
        # Extract edges based on similarity
        sources = []
        targets = []
        weights = []
        
        n_items = len(business_df)
        item_ids = business_df['item_id'].values
        
        # Track edges to avoid duplicates when creating bidirectional edges
        edge_set = set()
        
        for i in range(n_items):
            # Get similarities for item i (exclude self-similarity)
            item_sims = similarities[i].copy()
            item_sims[i] = -1  # Exclude self-similarity
            
            # Get top-k most similar items (following IAGCF paper - no min_similarity threshold)
            top_indices = np.argsort(item_sims)[-top_k:][::-1]  # Descending order
            
            for j in top_indices:
                sim_score = item_sims[j]
                if sim_score > 0:  # Only require positive similarity
                    # Create bidirectional edges (avoid duplicates)
                    edge_pair = tuple(sorted([item_ids[i], item_ids[j]]))
                    if edge_pair not in edge_set:
                        edge_set.add(edge_pair)
                        
                        # Add bidirectional edges
                        sources.append(item_ids[i])
                        targets.append(item_ids[j])
                        weights.append(sim_score)
                        
                        sources.append(item_ids[j])
                        targets.append(item_ids[i])
                        weights.append(sim_score)
        
        if sources and targets:
            edge_index = torch.stack([
                torch.LongTensor(sources),
                torch.LongTensor(targets)
            ])
            edge_weight = torch.FloatTensor(weights)
            
            # Statistics
            unique_pairs = len(set(tuple(sorted([s, t])) for s, t in zip(sources, targets))) // 2
            avg_similarity = sum(weights) / len(weights)
            min_weight = min(weights)
            max_weight = max(weights)
            
            print(f"Created {edge_index.shape[1]} directed edges ({unique_pairs} unique pairs)")
            print(f"Similarity weights: min={min_weight:.3f}, avg={avg_similarity:.3f}, max={max_weight:.3f}")
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weight = torch.empty(0, dtype=torch.float)
            print("No item-item connections created")
        
        return edge_index, edge_weight
    
    def _prepare_item_features(self, business_df):
        """
        Prepare item features by encoding city and categories
        Returns: feature matrix (dense for cosine similarity)
        """
        print("Preparing item features (city + categories)...")
        
        # Encode cities
        city_encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
        city_encoded = city_encoder.fit_transform(business_df[['city']])
        
        # Prepare categories
        categories_list = []
        for cats in business_df['categories']:
            if pd.isna(cats) or cats == 'None':
                categories_list.append([])
            else:
                categories_list.append(str(cats).split(', '))
        
        # Encode categories
        mlb = MultiLabelBinarizer(sparse_output=True)
        categories_encoded = mlb.fit_transform(categories_list)
        
        # Combine features
        item_features = hstack([city_encoded, categories_encoded])
        
        # Convert to dense for cosine similarity computation
        if hasattr(item_features, 'toarray'):
            # It's a sparse matrix
            print(f"Item features shape: {item_features.shape}")
            print(f"Feature density: {item_features.nnz / (item_features.shape[0] * item_features.shape[1]):.4f}")
            item_features = item_features.toarray()
        else:
            # It's already dense
            print(f"Item features shape: {item_features.shape}")
            non_zero = np.count_nonzero(item_features)
            print(f"Feature density: {non_zero / (item_features.shape[0] * item_features.shape[1]):.4f}")
        
        return item_features


    def combine_friendship_trust_edges(self, friend_edges, friend_weights, trust_edges, trust_weights, gamma=0.5):
        """Combine friendship and trust edges using weighted formula:
        - If both exist: weight(A,B) = γ × friendship(A,B) + (1-γ) × trust(A,B)
        - If only one exists: keep original weight
        """
        # Create a dictionary to store combined edges
        edge_dict = {}
        print(f"Combining edges with gamma={gamma}...")
        # Add friendship edges
        for i in range(friend_edges.shape[1]):
            u, v = friend_edges[0, i].item(), friend_edges[1, i].item()
            edge_key = (u, v)
            edge_dict[edge_key] = {'friend': friend_weights[i].item(), 'trust': None}

        # Add trust edges
        for i in range(trust_edges.shape[1]):
            u, v = trust_edges[0, i].item(), trust_edges[1, i].item()
            edge_key = (u, v)
            if edge_key in edge_dict:
                edge_dict[edge_key]['trust'] = trust_weights[i].item()
            else:
                edge_dict[edge_key] = {'friend': None, 'trust': trust_weights[i].item()}

        # Combine with weighted formula
        sources = []
        targets = []    
        weight = []

        for (u, v), weights in edge_dict.items():
            if weights['friend'] is not None and weights['trust'] is not None:
                # Both exist: use weighted combination
                combined_weight = gamma * weights['friend'] + (1 - gamma) * weights['trust']
            elif weights['friend'] is not None:
                # Only friendship exists: keep original weight
                combined_weight = weights['friend']
            else:
                # Only trust exists: keep original weight
                combined_weight = weights['trust']
            sources.append(u)  # user_a → user_b
            targets.append(v)
            weight.append(combined_weight)


        
        if sources and targets :
            # Build edge_index like UI edges
            edge_index = torch.stack([
                torch.LongTensor(sources),  # Row 0: Sources
                torch.LongTensor(targets)   # Row 1: Targets
            ])
            edge_weight = torch.FloatTensor(weight)
            

        return edge_index, edge_weight    
