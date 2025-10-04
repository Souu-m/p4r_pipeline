# Item Profile Generation and Embedding Pipeline
## This repository provides an unofficial implementation of the profile generation and embedding methodology described in the research paper "A Prompting-Based Representation Learning Method for Recommendation with Large Language Models" (P4R).
The pipeline follows the paper's core two-stage approach:

Profile Generation: A Large Language Model (LLM) transforms raw item metadata (e.g., name,city, rating,review_count..) into rich, narrative Item Profiles, creating semantically meaningful textual features.

Embedding & Projection: The textual profiles are converted into high-dimensional vectors (embeddings), which are then projected into compact, optimized features for the final recommendation model.

- Repository Structure and Data Flow
The project is divided into two distinct processing stages, 
each with its own folder, reflecting the recommended code split.


1- Profile Generation Folder (p4r_profile_generation/)
*****STAGE 1****

p4r_profile_generation/
├── src/
│   ├── generate_item_profiles.py # The script that takes metadata and generate profiles of it
│   └── __init__.py               # LLM Initialization and API calls
├── outputs/
│   └── business_profiles.json    # LLM-Generated Text Profiles (Input for Stage 2)
└── business_df.csv             #  INPUT DATA (Raw Item Metadata)

*****STAGE 2****
p4r_encode_text_to_embedding/
├── embeddings_model.py # Converts the textual Item Profiles (from stage1), 
│                       # into high-dimensional (D=768) semantic vector embeddings using the Gemini API embedding model.
│  
│   
├── projection.py   # Reduces the dimensionality of the high-dimensional embeddings to a smaller vector size, typically for memory and efficiency in the final recommendation model.
│                   # uses a Fully Connected (FC) layer with an optional ReLU activation for optimization.
└── business_profiles.json          #  INPUT DATA (LLM-Generated Text Profiles FROM STAGE1)


Usage and Execution

Phase 1: Item Profile Generation
This step generates the textual feature descriptions using the LLM.

Ensure your raw data file, business_df.csv, is placed within the p4r_profile_generation/ directory.

CRITICAL Data Requirement: The input DataFrame, business_df, must be sorted by original_item_id before processing to maintain a consistent mapping between items and profiles/embeddings throughout the pipeline.

Execute the profile generation script:


Phase 2: Embedding Encoding
This step converts the generated text profiles into high-dimensional vectors.

The script encode_embeddings.py loads the text profiles from the output of Phase 1.

It uses an external model (GEMINI TEXT EMBEDDING) to convert the text fields into vectors.


Phase 3: Dimensionality Reduction (Projection)
This step optimizes the vectors for the final model by reducing their size.

The script projection.py loads the high-dimensional vectors.

It applies the projection technique to obtain the final, smaller embeddings for the recommender model.
