# FindMyFund

FindMyFund is a semantic search system for financial securities, enabling users to query mutual funds, stocks, and holdings using natural language (e.g., "sbi debt funds"). Powered by a fine-tuned `all-MiniLM-L6-v2` model, FAISS for efficient indexing, and a Flask web interface, it delivers relevant results with a memory cache for instant recall of recent queries.

## Features
- **Semantic Search**: Matches queries against 16,481 mutual funds, 6,136 stocks, and 870,397 holdings.
- **Memory Cache**: Stores up to 100 recent queries for faster repeat searches.
- **Modern UI/UX**: Responsive Flask web app with card-based results, loading spinner, and cache-clearing option.
- **Optimized Indexing**: Uses FAISS (`IndexFlatIP` for funds/stocks, `IndexIVFFlat` for holdings) for quick similarity searches.
- **Dynamic Results**: Shows top 5 results per category with similarity scores and detailed metadata.

## Project Structure
FindMyFund/
├── data/
│   ├── raw/                    # Raw JSON data (mutual_funds_data.json, stock_data.json, mf_holdings_data.json)
│   ├── processed/              # Preprocessed CSVs (mutual_funds.csv, stocks.csv, holdings.csv, queries.csv)
│   ├── embeddings/             # Embeddings (.npy) and metadata (.csv)
│   └── index/                  # FAISS indexes (.faiss) and metadata (.csv)
├── models/
│   ├── finetuned_minilm/       # Initial fine-tuned model (1 epoch)
│   ├── finetuned_minilm_v2/    # Improved model (3 epochs)
│   └── finetuned_minilm_v3/    # Latest model (5 epochs with validation)
├── src/
│   ├── preprocess.py           # Converts JSON to CSVs and generates queries
│   ├── finetune.py             # Fine-tunes MiniLM model
│   ├── generate_embeddings.py  # Creates embeddings for metadata
│   ├── build_index.py          # Builds FAISS indexes
│   └── app.py                  # Flask web app with UI
├── .env/                       # Virtual environment
├── requirements.txt            # Python dependencies
└── README.md                   # This file


## Prerequisites
- Python 3.9 or higher
- ~10GB disk space for data, embeddings, and models
- NVIDIA GPU (optional, for faster training/indexing with CUDA)

## Installation
1. **Clone Repository** (if hosted):
   ```bash
   git clone <repository-url>
   cd FindMyFund
   
2. **Set Up Virtual Environment:**
python -m venv .env
.env\Scripts\activate  # Windows

3. **Install Dependencies:requirements.txt with:**
4. **Unzip holdings_index.faiss.zip**
5. **Launch Web App:  python src/app.py**
6. **Feature:** Search:  Enter queries like "sbi debt funds" or "materials stocks".
View top 5 results per category (mutual funds, stocks, holdings) in a card-based layout.
Use the "Clear Cache" button to reset memory.

   Memory Feature:
Repeated queries return cached results instantly, marked as "(From Cache)".

7. **Example Queries**
Mutual Funds: "sbi debt funds" → Matches funds like "SBI FMP-78-1170D(IDCW)-Direct Plan".
Stocks: "materials stocks" → Matches stocks like "Sahaj Fashions Ltd.".
Holdings: "funds with can fin homes holdings" → Matches relevant holdings.
Current Status
Strengths:
Fast searches (<1 second, instant for cached queries).
Responsive UI with dynamic features (loading spinner, result counts).
Robust pipeline from raw JSON to search results.
Ongoing Improvements:
Similarity scores currently low (<0.5); targeting >0.7 via enhanced training.
Ensuring diverse results per category (addressed duplicate result issue).
Next Steps:
Refine synthetic queries for better model training.
Explore larger model (all-MiniLM-L12-v2) if needed.
