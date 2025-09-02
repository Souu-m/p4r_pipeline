# Semantic Aspects Aware Review Extraction

This project helps extract semantic aspects from product reviews using natural language processing.

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/bry1ni/semantic_aspects_aware_review_extraction.git
   cd semantic_aspects_aware_review_extraction
   ```

2. **Create and activate a virtual environment**
   ```bash
   # Using uv (recommended)
   uv sync # This will automatically create and activate venv + install dependencies

   # Using pip
   python -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # OR
   .venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**
   ```bash
   # if you already used `uv sync` no need to run this
   # Using pip
   pip install -r requirements.txt
   ```

## Project Structure

```
.
├── src/
│   ├── utils.py         # Utility functions
│   ├── step1.py         # 1st step for the model
│   ├── nettoyage2.py     # cleaning the vocab
│   ├── step2.py         # 2nd step for the model
│   └── data.csv         # Sample product reviews data
├── requirements.txt     # Project dependencies
└── README.md           # This file
```

## How to Use

1. **Prepare your data**
   - Place your product reviews in `src/data.csv`
   - The CSV should have columns: `product_id`, `product_name`, `review` and `result`

2. **Run the project**
   ```bash
   python src/step1.py # you will get list_aspect_raw.csv 
   python src/nettoyage2.py # you will get vocab_with_freq.csv you should see it and decide your top-k aspect , go inside the code and modifie top-k= .. and run again python src/nettoyage2.py to get vocab_final.txt
   python src/step2.py # you will get the final result reviews_aspect_positive_only.csv
   ```

3. **Check results**
   - Results will be saved in `outputs folder` in the project root directory

## Example Data Format

Your `data.csv` should look like this:
```csv
product_id,product_name,review
P001,Wireless Headphones,"Great sound quality and comfortable fit!"
P002,Smart Watch,"Battery life could be better but features are good."
```

## Troubleshooting

If you encounter any issues:

1. Make sure all dependencies are installed correctly
2. Check that your data.csv file is in the correct format
3. Ensure you're running the script from the project root directory

## Contributing

Feel free to submit issues and enhancement requests!
