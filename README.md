# Item Profile Generation and Embedding Pipeline
## This repository provides an unofficial implementation of the profile generation and embedding methodology described in the research paper "A Prompting-Based Representation Learning Method for Recommendation with Large Language Models" (P4R).
The pipeline follows the paper's core two-stage approach:

Profile Generation: A Large Language Model (LLM) transforms raw item metadata (e.g., name,city, rating,review_count..) into rich, narrative Item Profiles, creating semantically meaningful textual features.

Embedding & Projection: The textual profiles are converted into high-dimensional vectors (embeddings), which are then projected into compact, optimized features for the final recommendation model.

- Repository Structure and Data Flow

The project is divided into two distinct processing stages, each with its own folder, reflecting the recommended code split.

### Stage 1: Profile Generation (`p4r_profile_generation/`)

| File/Folder | Purpose | I/O |
| :--- | :--- | :--- |
| **`business_df.csv`** | **INPUT DATA:** Raw item metadata. | In |
| `src/generate_item_profiles.py` | Main script orchestrating LLM calls for profile generation. | Code |
| `src/__init__.py` | Handles LLM Initialization and API connectivity. | Code |
| `outputs/business_profiles.json` | **OUTPUT:** LLM-generated Item Profiles (Input for Stage 2). | Out |

***

### Stage 2: Embedding and Projection (`p4r_textual_embedding/`)

| File/Folder | Purpose | Details |
| :--- | :--- | :--- |
| **`business_profiles.json`** | **INPUT:** Textual Item Profiles from Stage 1. | In |
| `encode_profiles.py` | Converts textual profiles into **high-dimensional semantic vectors** ($\text{D}=768$). | Code |
| `projection.py` | **Reduces the dimensionality** of the embeddings. Uses an FC layer with optional $\text{ReLU}$ activation. | Code |
| `item_embeddings.pt` | **OUTPUT:** Final, projected vectors for the recommender model. | Out |


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
