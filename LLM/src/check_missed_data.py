import pandas as pd


business_df = pd.read_csv(r"LLM\src\business_df.csv")


missing_data = business_df[[
    "name", "city", "state", "latitude", "longitude", 
    "stars", "review_count", "attributes", "categories", "original_item_id"
]].isnull().sum()

print(missing_data)    
    
