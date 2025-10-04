import os
from google import genai
from google.genai import types
import numpy as np
import torch
import json
from concurrent.futures import ThreadPoolExecutor
import time

from dotenv import load_dotenv

# Load .env and configure Gemini API key
load_dotenv()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

def load_item_profiles(json_file_path, limit=None):
   """Load item profiles from JSON file with optional limit"""
   with open(json_file_path, 'r') as f:
       data = json.load(f)
       if limit:
           return data[:limit]
       return data

def process_batch(batch_data, batch_start_idx):
    """Process a single batch of items"""
    item_ids = [item['item_original_id'] for item in batch_data]
    item_profiles = [item['item_profile'] for item in batch_data]
    
    try:
        # Get embeddings for the batch
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=item_profiles,
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
        
        # Convert to PyTorch float tensors and pair with IDs
        batch_results = {}
        for i, embedding in enumerate(result.embeddings):
            item_id = item_ids[i]
            batch_results[item_id] = torch.tensor(embedding.values, dtype=torch.float32)
            
        print(f"Processed batch starting at index {batch_start_idx} ({len(batch_data)} items)")
                
        # Add a delay to prevent rate limiting
        time.sleep(2)  # Wait for 2 seconds after each batch
        
        return batch_results
        
    except Exception as e:
        print(f"Error processing batch starting at {batch_start_idx}: {e}")
        return {}

def encode_item_profiles(json_file_path, batch_size=5, max_workers=4):
    """
    Encode item profiles into embeddings with batching and multithreading
    
    Args:
        json_file_path: Path to JSON file with item profiles
        batch_size: Number of items to process per batch
        max_workers: Number of concurrent threads
    
    Returns:
        Dictionary mapping item_original_id to PyTorch float tensors
    """
    # Load data
    items = load_item_profiles(json_file_path)
    print(f"Loaded {len(items)} items from {json_file_path}")
    
    # Create batches
    batches = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batches.append((batch, i))
    
    print(f"Processing {len(batches)} batches with batch_size={batch_size}, max_workers={max_workers}")
    
    # Process batches in parallel
    all_embeddings = {}
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batch processing tasks
        future_to_batch = {
            executor.submit(process_batch, batch_data, batch_start): batch_start 
            for batch_data, batch_start in batches
        }
        
        # Collect results as they complete
        for future in future_to_batch:
            batch_results = future.result()
            all_embeddings.update(batch_results)
    
    end_time = time.time()
    print(f"Completed processing {len(all_embeddings)} items in {end_time - start_time:.2f} seconds")
    
    return all_embeddings


if __name__ == "__main__":
    # Process your item profiles
    embeddings = encode_item_profiles('business_profiles.json', batch_size=4, max_workers=3)
    
    # Display a sample item's embedding (ID, shape, and first few values)
    if embeddings:
        sample_id = list(embeddings.keys())[0]
        print(f"\nSample embedding for item {sample_id}:")
        print(f"Shape: {embeddings[sample_id].shape}")
        print(f"First 5 values: {embeddings[sample_id][:5]}")
    
    # Save embeddings (PyTorch format)
    torch.save(embeddings, 'item_embeddings.pt')