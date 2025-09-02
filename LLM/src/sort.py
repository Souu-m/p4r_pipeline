import pandas as pd


# Load your CSV
df = pd.read_csv("business_df.csv")

# Sort by the last column (original_item_id)
df_sorted = df.sort_values(by="original_item_id")
# Move 'original_item_id' to the front
cols = ["original_item_id"] + [col for col in df_sorted.columns if col != "original_item_id"]
df_sorted = df_sorted[cols]
# Save back to CSV (without index column unless you want it)
df_sorted.to_csv("business_df_sorted.csv", index=False)
