"""Configuration settings for the GCN recommendation system"""
import torch
class Config:

    # In config.py
    
    # Paths to the datasets for preprocessing.py script only, once the data is preprocessed and saved to Datasets/preprocessed, these paths are not used anymore
    BUSINESS_DATA_PATH = r"D:\PFE_CODE\PFE_Aspects\Datasets\yelp_dataset\yelp_academic_dataset_business_shopping.csv"
    REVIEW_DATA_PATH = r"D:\PFE_CODE\PFE_Aspects\Datasets\yelp_dataset\yelp_academic_dataset_review_shopping.csv"
    USER_DATA_PATH = r"D:\PFE_CODE\PFE_Aspects\Datasets\yelp_dataset\yelp_academic_dataset_user_shopping.csv"
   
    #BUSINESS_DATA_PATH = r"D:\PFE_CODE\PFE_Aspects\Datasets\YelpRestaurants\yelp_academic_dataset_business_restaurants.csv"
    #USER_DATA_PATH = r"D:\PFE_CODE\PFE_Aspects\Datasets\YelpRestaurants\yelp_academic_dataset_user_restaurants.csv"
    #REVIEW_DATA_PATH = r"D:\PFE_CODE\PFE_Aspects\Datasets\YelpRestaurants\yelp_academic_dataset_review_restaurants.csv"

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' #if available 
    SEED = 14 # Random seed for reproducibility
    # Aspect-specific configuration
    USE_ASPECT_EMBEDDINGS = False
    ENCODER = True # no need to use it because after experimenting it turns that adding my transfromer model with GCN perform better without transfromer encoder
    
    ALPHA=0.6

    # Transformer-specific configuration
    USE_TRANSFORMER = False  # Whether to use transformer layers ( the one that improve the performance of GCN when its enabled (it should be enabled with edge features))
    USE_EDGE_FEATURES = False  # Whether to use edge weights in transformer, if transformer is not used, this is ignored
    TRANSFORMER_LAYERS = 1
    TRANSFORMER_HEADS = 4
    
    # Graph configuration
    GRAPH_CONFIG = 'ui'  # Options: 'ui', 'ui_friend', 'ui_trust', 'ui_friend_trust'
    MIN_RATING = 1
    MIN_JACCARD = 0.0
    MIN_COMMON_ITEMS = 10
    MIN_SIMILARITY = 0.0
    MIN_COOCCUR = 3
    GAMMA = 0.9
    # Model configuration
    EMBEDDING_DIM = 64
    NUM_LAYERS = 1
    LEARNING_RATE = 0.01
    
    # Training configuration
    NUM_EPOCHS = 50
    PATIENCE = 5
    BATCH_SIZE = 500
    TEST_SIZE = 0.1
    VAL_SIZE = 0.1
    Train_RATE = 0.8
    
    # Evaluation configuration
    k=20 # k for recall@k and precision@k top_k recommendations
    Threshold =3.5 # Threshold for binary classification recall@k and precision@k
    # Preprocessing
    KEEP_WITHOUT_REVIEWS = False
