
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

import os
import pickle
import torch
import numpy as np
import copy
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, accuracy_score



class Preprocessor:
    """
    Clean preprocessing pipeline that:
    1. Removes reviews with missing user_id, item_id, or rating
    2. Keeps latest review per user-item pair
    3. Optionally filters to only users/items with reviews
    4. Returns encoders for later use
    """
    
    def __init__(self):
        self.user_encoder = None
        self.item_encoder = None
        self.stats = {}
        
    def preprocess_all(self, user_df, business_df, review_df, 
                      keep_without_reviews=True,
                      user_id_col='user_id', 
                      item_id_col='item_id',
                      rating_col='rating'):
        """
        Main preprocessing function
        
        Args:
            user_df: User dataframe
            item_df: Item/business dataframe
            review_df: Review/interaction dataframe
            keep_without_reviews: If True, keeps users/items without reviews (when NOT using sentiment)
            user_id_col: Name of user ID column
            item_id_col: Name of item ID column
            rating_col: Name of rating column
            
        Returns:
            Tuple of (user_df, item_df, review_df, user_encoder, item_encoder)
        """
        
        print("keep_without_reviews",keep_without_reviews)
        # Step 1: Clean review data - remove invalid rows and keep latest rating
        review_df = review_df.rename(columns={'stars':'rating', 'business_id':'item_id'})
        item_df = business_df.rename(columns={'business_id': 'item_id'})
        


        # Remove reviews with missing critical data
        initial_review_count = len(review_df)
        review_df = review_df.dropna(subset=[user_id_col, item_id_col, rating_col])
        print(f"Removed {initial_review_count - len(review_df)} reviews with missing user_id, item_id, or rating")

        # Keep only latest review per user-item pair (if duplicates exist)
        review_df = review_df.sort_values(by=[user_id_col, item_id_col, 'date']).drop_duplicates(subset=[user_id_col, item_id_col], keep='last')
        review_df_processed = review_df.reset_index(drop=True)


        # Store original counts
        self.stats['original'] = {
            'users': len(user_df),
            'items': len(item_df),
            'reviews': len(review_df)
        }
        

        valid_users = set(review_df[user_id_col].unique())
        valid_items = set(review_df[item_id_col].unique())
        
        # Step 2: Filter users/items based on reviews (optional), 
        # means we can keep users/items that have no reviews in case we are not using sentiment analysis
        if keep_without_reviews:
            print("\nKeeping ALL users and items (no sentiment analysis)")
            user_df_processed = user_df[user_df[user_id_col].isin(valid_users)].copy()
            item_df_processed = item_df[item_df[item_id_col].isin(valid_items)].copy()
        else:
            print("\nFiltering to users/items with BOTH ratings AND review text")
            print("(Using this mode FOR sentiment analysis)")
            
            # For sentiment analysis, also check for non-empty review text
            if 'text' in review_df.columns:
                # Get IDs that have both ratings AND non-empty review text
                reviews_with_text = review_df[
                    (review_df['text'].notna()) & 
                    (review_df['text'].str.strip() != '') &
                    (review_df['text'] != 'None')
                ]
                valid_users_with_text = set(reviews_with_text[user_id_col].unique())
                valid_items_with_text = set(reviews_with_text[item_id_col].unique())
                
                user_df_processed = user_df[user_df[user_id_col].isin(valid_users_with_text)].copy()
                item_df_processed = item_df[item_df[item_id_col].isin(valid_items_with_text)].copy()
                
            else:
                print("Warning: No 'text' column found, using same filtering as non-sentiment mode")
                user_df_processed = user_df[user_df[user_id_col].isin(valid_users)].copy()
                item_df_processed = item_df[item_df[item_id_col].isin(valid_items)].copy()
        
        
        

        # Step 3: Encode IDs
        self._encode_ids(user_df_processed, item_df_processed, review_df_processed,
                        user_id_col, item_id_col)
        
        # Store final counts
        self.stats['final'] = {
            'users': len(user_df_processed),
            'items': len(item_df_processed),
            'reviews': len(review_df_processed)
        }
        
        # OFFSET item IDs to avoid overlap, 
        #item_id_global stand for the global item ID space (users + items)
        n_users = len(self.user_encoder.classes_)  
        # Keep original item_id and create item_id_global
        item_df_processed['original_item_id'] = item_df_processed[item_id_col]  # Save original
        item_df_processed['item_id'] = item_df_processed[item_id_col] + n_users  # Rename to item_id

        review_df_processed['original_item_id'] = review_df_processed[item_id_col]  # Save original  
        review_df_processed['item_id'] = review_df_processed[item_id_col] + n_users  # Rename to item_id
        # Print summary
        self._print_summary()
        
        return user_df_processed, item_df_processed, review_df_processed, self.user_encoder, self.item_encoder
    


    
    def _encode_ids(self, user_df, item_df, review_df, user_id_col, item_id_col):
        """
        Encode IDs consistently across all dataframes
        """
        print("\nStep 4: Encoding IDs")
        print("-" * 40)
        
        # Create encoders
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
        # Collect ALL unique IDs from all datasets
        all_user_ids = pd.concat([
            user_df[user_id_col],
            review_df[user_id_col]
        ]).dropna().unique()
        
        all_item_ids = pd.concat([
            item_df[item_id_col],
            review_df[item_id_col]
        ]).dropna().unique()
        
        # Fit encoders
        self.user_encoder.fit(all_user_ids)
        self.item_encoder.fit(all_item_ids)
        
        # Apply encoding
        user_df['user_id'] = self.user_encoder.transform(user_df[user_id_col])
        item_df['item_id'] = self.item_encoder.transform(item_df[item_id_col])
        review_df['user_id'] = self.user_encoder.transform(review_df[user_id_col])
        review_df['item_id'] = self.item_encoder.transform(review_df[item_id_col])
        
        print(f"Encoded {len(self.user_encoder.classes_)} unique users")
        print(f"Encoded {len(self.item_encoder.classes_)} unique items")
    
    def _print_summary(self):
        """
        Print preprocessing summary
        """
        print("\n" + "="*60)
        print("PREPROCESSING SUMMARY")
        print("="*60)
        
        print("\nData Reduction:")
        print(f"Users: {self.stats['original']['users']} → {self.stats['final']['users']}")
        print(f"Items: {self.stats['original']['items']} → {self.stats['final']['items']}")
        print(f"Reviews: {self.stats['original']['reviews']} → {self.stats['final']['reviews']}")
        
        print("\nEncoders saved for later use:")
        print("- user_encoder: Can map user IDs back to original")
        print("- item_encoder: Can map item IDs back to original")


def preprocess_and_save(config, save_dir="Project/Datasets/sentiment"):
    """
    Fonction principale pour preprocesser et sauvegarder les données
    """
    print("=== Starting preprocessing and saving ===")
    
    # Créer le dossier s'il n'existe pas
    os.makedirs(save_dir, exist_ok=True)
    
    # Charger les données brutes 
    from util.data_utils import load_data  
    user_df_raw, business_df_raw, review_df_raw =  load_data(config)
    
    # Preprocesser
    preprocessor = Preprocessor()
    user_df, business_df, review_df, user_encoder, item_encoder = preprocessor.preprocess_all(
        user_df_raw, business_df_raw, review_df_raw,config.KEEP_WITHOUT_REVIEWS,
    )
    
    # Sauvegarder les DataFrames
    user_df.to_csv(os.path.join(save_dir, "user_df.csv"), index=False)
    business_df.to_csv(os.path.join(save_dir, "business_df.csv"), index=False)
    review_df.to_csv(os.path.join(save_dir, "review_df.csv"), index=False)
    
    # Sauvegarder les encoders
    with open(os.path.join(save_dir, "user_encoder.pkl"), 'wb') as f:
        pickle.dump(user_encoder, f)
    with open(os.path.join(save_dir, "item_encoder.pkl"), 'wb') as f:
        pickle.dump(item_encoder, f)
    
    print(f"✅ Données preprocessées sauvegardées dans {save_dir}")
    return user_df, business_df, review_df, user_encoder, item_encoder

if __name__ == "__main__":
    from config.config import Config
    config = Config()
    preprocess_and_save(config)