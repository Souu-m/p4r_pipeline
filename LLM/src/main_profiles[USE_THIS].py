import re
import pandas as pd
import json
from tqdm import tqdm
from __init__ import chat_llm
from concurrent.futures import ThreadPoolExecutor


PROMPT_BUSINESS_PROFILE_BATCH = (
    "You will serve as an assistant to analyze which types of users will like this business. "
    "The intrinsic knowledge is: name: the name of the business. categories: categories of the business. "
    "The extrinsic knowledge is: stars: average rating from users who have used this business, review counts.\n"
    "Like these businesses. For each business, provide analysis in JSON format.\n\n"
    "Here are multiple businesses to analyze:\n<businesses>\n\n"
    "Note: the businesses block will include 'average_rating' and 'review_count' already as exact natural-language phrases "
    "(e.g. 'excellent','well-rated','moderately rated','mixed/low','poorly rated','not available' and "
    "'very popular','relatively popular','somewhat established','new/rarely reviewed','no reviews available'). "
    "Use those PROVIDED phrase tokens verbatim in your output; do NOT invent, convert, or paraphrase them.\n"
    "For EACH business, provide analysis in this exact JSON format:\n"
    "[\n"
    "  {\n"
    '    "business_id": "<original_item_id>",\n'
    '    "summarization": "what is this business for...",\n'
    '    "user_preferences": "Either extracted or predicted according to the provided information...",\n'
    '    "recommendation_reasoning": "give explanation and reasoning for why these users would enjoy the business using complete sentences. Refer to rating and review count using only the prescribed natural-language phrases."\n'
    "  }\n"
    "]\n\n"
    "CRITICAL: Return ONLY a valid JSON array. No markdown, no extra text. DO NOT add trailing commas.\n"
    "Rules:\n"
    "  - Do NOT invent or add any attributes (e.g., 'premium,' 'affordable,' 'luxury,' 'high-quality,' 'new location') that are not explicitly present in the metadata. "
    "Be strict about hallucinations, ensuring user_preferences and recommendation_reasoning only reflect categories, location, and provided stars/review_count phrases without speculative descriptors.\n"
    "  - Replace business_id with the actual business number.\n"
    "  - Summarization must mention the business name and location, Always list categories exactly as provided, separated only by commas (no “and” before the last category). " 
    "(e.g., '[Name] in [City, State] offers [categories].'). Use a clear verb like 'offers' or 'provides' to ensure the sentence is grammatically complete and flows naturally.\n"
    "  - Always include all categories from the metadata exactly as provided (verbatim, not paraphrased). Use each category once, separated by commas only (no “and”)."
    "If categories are missing, write 'categories not available'.\n"
    "  - For 'user_preferences': infer user types only from categories and location. Do not add attributes such as price, quality, or brand reputation unless present in the metadata.\n"
    "  - For 'recommendation_reasoning': ALWAYS start with 'Based on the categories listed ([comma-separated categories]) and the location in [city, state], it is likely that [inferred user types]...'. "
    "Then, in the next sentence, restate the stars phrase and review_count phrase exactly as provided (e.g., 'The business is moderately rated and somewhat established.'). "
    "Do not interpret or evaluate these phrases (e.g., avoid 'meets expectations,' 'may not satisfy customers,' or similar). "
    "If both are 'not available' and 'no reviews available,' state that recommendations rely only on categories and location, without implying customer sentiment.\n"
    "  - Output must be valid JSON using double quotes for all keys and string values. Preserve apostrophes inside strings but ensure they are properly JSON-escaped if necessary.\n\n"
    "Examples:\n"
    "  - With ratings: recommendation_reasoning: 'Based on the categories listed (Ice Cream & Frozen Yogurt, Sandwiches, Food, American (Traditional), Restaurants) and the location in Scottsdale, AZ, "
    "it is likely that families with children, couples looking for a casual dining experience, and dessert enthusiasts would enjoy this business. "
    "The business is moderately rated and somewhat established.'\n"
    "  - Missing ratings: recommendation_reasoning: 'Based on the categories listed (Coffee & Tea, Food) and the location in Seattle, WA, it is likely that coffee lovers, remote workers, and casual meetup groups would enjoy this business. "
    "The not available status and no reviews available mean that recommendations rely only on categories and location.'\n"
)


BATCH_SIZE = 5 
MAX_WORKERS = 3

#***********************************************************************************************#
#***********************************************************************************************#
#***********************************************************************************************#
import math
import pandas as pd
import numpy as np
import re

def is_missing_value(v):
    """Return True if value v should be treated as missing.
    Handles None, numpy/pandas NaN, empty strings, whitespace-only, and common textual tokens.
    """
    # pandas/np missing
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except Exception:
        pass

    # Strings that represent missing
    s = str(v).strip()
    if s == "":
        return True
    if s.lower() in {"none", "nan", "null", "na", "n/a", "not available", "missing"}:
        return True

    return False





#***********************************************************************************************#
#***********************************************************************************************#
#***********************************************************************************************#
#***********************************************************************************************#

def clean_text_for_json(text):
    if pd.isna(text):
        return ""
    text = str(text)
    
    # Fix corrupted Unicode sequences first
    text = re.sub(r'\\x[0-9a-fA-F]{2}', '', text)  # Remove \xXX sequences
    
    # Handle special characters that break JSON
    text = text.replace('\\', '\\\\')  # Escape backslashes first
    text = text.replace('"', '\\"')   # Escape double quotes
    # Remove non-printable characters except basic punctuation
    text = re.sub(r'[^\x20-\x7E]', '', text)
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def fix_json_structure(text):
    """Fix common JSON structural issues"""
    # Remove trailing commas before closing braces/brackets
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # Fix missing commas between objects in arrays
    text = re.sub(r'}(\s*){', r'},\1{', text)
    
    # Fix missing quotes around property names
    text = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)
    
    return text

def fix_json_escapes(text):
    """Fix common JSON escape issues from LLM responses"""
    # Remove escape before single quotes
    text = text.replace("\\'", "'")
    text = text.replace("\\&", "&")
    text = text.replace("\\#", "#")
    
    # Fix bad unicode escapes like \u' or \u:
    text = re.sub(r'\\u(?![0-9a-fA-F]{4})', 'u', text)  # keep only valid \uXXXX

    # Fix any remaining invalid escapes (keep only valid JSON escapes)
    text = re.sub(r'\\([^"\\\/bfnrtu])', r'\1', text)

    return text

# Global list to track fallback per business
fallback_used_ids = []
def robust_json_parse(response_text,business_ids=None):
    """Robust JSON parsing with fallback strategies (enhanced)"""
    response_text = response_text.strip()
    
    # Remove markdown code blocks if present
    if response_text.startswith('```'):
        lines = response_text.split('\n')
        response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
        response_text = response_text.strip()
    
    # New: Handle if wrapped in a dict like {"json": [...]}
    if response_text.startswith('{') and '"json":' in response_text:
        try:
            data = json.loads(response_text)
            if 'json' in data and isinstance(data['json'], list):
                response_text = json.dumps(data['json'])
        except:
            pass
    
    # Apply all fixes
    response_text = fix_json_escapes(response_text)
    response_text = fix_json_structure(response_text)
    
    # Strategy 1: Direct parsing
    try:
        return json.loads(response_text)        
    except json.JSONDecodeError as e:
        print("Direct JSON parse failed")
        print("Error:", e)
        print("Offending text snippet:", response_text[max(0, e.pos-50):e.pos+50])
    
    # Strategy 2: Extract and fix JSON array
    try:
        # Find the JSON array pattern with more flexible matching
        array_pattern = r'\[\s*\{.*?\}\s*\]'
        match = re.search(array_pattern, response_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            json_str = fix_json_escapes(json_str)
            json_str = fix_json_structure(json_str)
            
            
            if business_ids:
                # Mark fallback for all business_ids in this response
                fallback_used_ids.extend(business_ids)   
            return json.loads(json_str)   
    except Exception as e:
        print(f"Regex extraction failed: {e}")
    
    # Strategy 3: Manual reconstruction attempt
    try:
        # Try to extract individual business objects and reconstruct
        object_pattern = r'\{\s*"business_id"[^}]+\}'
        objects = re.findall(object_pattern, response_text, re.DOTALL)
        
        if objects:
            reconstructed = '[' + ','.join(objects) + ']'
            reconstructed = fix_json_escapes(reconstructed)
            reconstructed = fix_json_structure(reconstructed)
            # Track fallback
            if business_ids:
                fallback_used_ids.extend(business_ids)
            return json.loads(reconstructed)
    except Exception as e:
        print(f"Manual reconstruction failed: {e}")
    
    print(f"All JSON parsing failed. Response preview: {response_text[:200]}...")
    if business_ids:
        fallback_used_ids.extend(business_ids)
    return None

def process_business_batch(batch_data):
    """Process a batch of businesses for profile generation"""
    batch_df, start_idx = batch_data
    
    try:
        # Clean business data for prompt
        businesses_str = "\n\n".join([
            f"Business {row['original_item_id']}:\n"
            f"- Name: {clean_text_for_json(row['name'])}\n"
            f"- Categories: {clean_text_for_json(row['categories'])}\n"
            f"- City: {clean_text_for_json(row['city'])}, State: {clean_text_for_json(row['state'])}\n"
            f"- average_rating: {clean_text_for_json(row['stars'])}\n"
            f"- review_count: {clean_text_for_json(row['review_count'])}\n"
            for _, row in batch_df.iterrows()
        ])

        
        prompt = PROMPT_BUSINESS_PROFILE_BATCH.replace("<businesses>", businesses_str)
        
        # Get LLM response
        resp = chat_llm(prompt, max_tokens=512 * len(batch_df))
        
        # Parse JSON response
        parsed_data = robust_json_parse(resp,business_ids=batch_df['original_item_id'].tolist())
        
        if parsed_data is None or not isinstance(parsed_data, list):
            print(f"JSON parsing failed for batch {start_idx}")
            parsed_data = []
        else:
            print(f"Successfully parsed {len(parsed_data)} business profiles from batch {start_idx}")
        
        # Process results
        batch_results = []
        for i, (_, row) in enumerate(batch_df.iterrows()):
            if i < len(parsed_data) and isinstance(parsed_data[i], dict):
                result = parsed_data[i]
                summarization = clean_text_for_json(result.get('summarization', ''))
                user_preferences = clean_text_for_json(result.get('user_preferences', ''))
                reasoning = clean_text_for_json(result.get('recommendation_reasoning', ''))
            else:
                print(f"Missing/invalid data for business {start_idx + i + 1}")
                summarization = user_preferences = reasoning = ""
            
            # Create item_profile
            item_profile = (
                f"[Summarization] {summarization} "
                f"[User Preferences] {user_preferences} "
                f"[Recommendation Reasoning] {reasoning}"
            )

            batch_results.append({
                'summarization': summarization,
                'user_preferences': user_preferences,
                'recommendation_reasoning': reasoning,
                'item_profile': item_profile,
                'original_item_id': row['original_item_id']
            })
        
        return batch_results
        
    except Exception as e:
        print(f"Error processing batch starting at index {start_idx}: {e}")
        # Return empty results for the batch
        empty_profile = "[Summarization]  [User Preferences]  [Recommendation Reasoning] "
        return [{'summarization': '', 'user_preferences': '', 'recommendation_reasoning': '', 
                'item_profile': empty_profile, 'original_item_id': row['original_item_id']} 
                for _, row in batch_df.iterrows()]

def generate_business_profiles_parallel(business_df, batch_size=BATCH_SIZE, max_workers=MAX_WORKERS):
    """Process businesses in parallel batches to generate profiles"""
    business_df = business_df.copy()
    
    # Prepare batch data
    batch_data_list = []
    for start in range(0, len(business_df), batch_size):
        batch = business_df.iloc[start:start+batch_size]
        batch_data_list.append((batch, start))
    
    # Process batches in parallel
    all_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(batch_data_list), desc="Generating business profiles") as pbar:
            batch_results = executor.map(process_business_batch, batch_data_list)
            for result in batch_results:
                all_results.extend(result)
                pbar.update(1)
    
    # Assign results back to dataframe
    for col in ['summarization', 'user_preferences', 'recommendation_reasoning', 'item_profile']:
        business_df[col] = [result[col] for result in all_results]
    
    # Add original_item_id column
    business_df['original_item_id'] = [result['original_item_id'] for result in all_results]
    
    return business_df

# Main execution
if __name__ == "__main__":
    print("Generating business profiles...")
    # IMPORTANT: Sort original item_id from business_df to ensure consistent batching
    business_df = pd.read_csv(r"business_df_sorted.csv")
    #business_df = business_df_df.sample(n=10, random_state=500).reset_index(drop=True)
    # Process business data to generate profiles
    business_df_with_profiles = generate_business_profiles_parallel(business_df)
    
    # Save CSV with all columns
    df_final = business_df_with_profiles[[
        "name", "city", "state", "latitude", "longitude", 
        "stars", "review_count", "categories",
        "summarization", "user_preferences", "recommendation_reasoning", "item_profile", "original_item_id"
    ]]
    
    df_final.to_csv("LLM/outputs/business_profiles.csv", index=False)
    print("Business profiles saved to: LLM/outputs/business_profiles.csv")
    
    # Create JSON output
    json_data = [
        {
            "item_original_id": row['original_item_id'],
            "item_profile": row['item_profile']
        }
        for _, row in business_df_with_profiles.iterrows()
    ]
    
    with open("LLM/outputs/business_profiles.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print("JSON profiles saved to: LLM/outputs/business_profiles.json")
    
    # Print sample results and token statistics
    print(f"\nProcessed {len(df_final)} businesses")
    
    # Token statistics
    estimated_tokens = [len(profile) // 4 for profile in df_final['item_profile']]
    over_512 = sum(1 for tokens in estimated_tokens if tokens > 512)
    avg_tokens = sum(estimated_tokens) / len(estimated_tokens) if estimated_tokens else 0
    max_tokens = max(estimated_tokens) if estimated_tokens else 0
    
    print(f"Token Statistics: Avg={avg_tokens:.1f}, Max={max_tokens}, Over 512 tokens={over_512}/{len(df_final)}")
    
    print("\nSample business profile:")
    if len(df_final) > 0:
        sample = df_final.iloc[0]
        print(f"Business: {sample['name']}")
        print(f"Original ID: {sample['original_item_id']}")
        print(f"Profile: {sample['item_profile'][:200]}...")
        print(f"Estimated tokens: {len(sample['item_profile']) // 4}")