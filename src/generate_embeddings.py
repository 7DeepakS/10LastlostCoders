# src/generate_embeddings.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import os
import time

def load_data():
    """Load preprocessed CSVs."""
    start_time = time.time()
    mf_df = pd.read_csv('data/processed/mutual_funds.csv')
    stock_df = pd.read_csv('data/processed/stocks.csv')
    holdings_df = pd.read_csv('data/processed/holdings.csv')
    print(f"Loaded mutual funds: {len(mf_df)} records")
    print(f"Loaded stocks: {len(stock_df)} records")
    print(f"Loaded holdings: {len(holdings_df)} records")
    print(f"Data loaded in {time.time() - start_time:.2f} seconds")
    return mf_df, stock_df, holdings_df

def generate_embeddings(model, texts, batch_size=32):
    """Generate embeddings for a list of texts."""
    start_time = time.time()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    print(f"Generated embeddings for {len(texts)} texts in {time.time() - start_time:.2f} seconds")
    return embeddings

def process_and_save_embeddings():
    """Generate and save embeddings for all metadata."""
    start_time = time.time()
    # Load fine-tuned model
    model = SentenceTransformer('models/finetuned_minilm_v4')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = model.to(device)

    # Load data
    mf_df, stock_df, holdings_df = load_data()

    # Extract metadata
    mf_metadata = mf_df['metadata'].tolist()
    stock_metadata = stock_df['metadata'].tolist()
    holdings_metadata = holdings_df['metadata'].tolist()

    # Generate embeddings
    print("Generating mutual fund embeddings...")
    mf_embeddings = generate_embeddings(model, mf_metadata)
    print("Generating stock embeddings...")
    stock_embeddings = generate_embeddings(model, stock_metadata)
    print("Generating holdings embeddings...")
    holdings_embeddings = generate_embeddings(model, holdings_metadata)

    # Save embeddings and metadata
    output_dir = 'data/embeddings/'
    os.makedirs(output_dir, exist_ok=True)

    # Mutual funds
    np.save(os.path.join(output_dir, 'mutual_funds_embeddings.npy'), mf_embeddings)
    mf_df[['schemeCode', 'schemeName', 'amcName', 'metadata']].to_csv(
        os.path.join(output_dir, 'mutual_funds_metadata.csv'), index=False
    )
    print("Saved mutual fund embeddings and metadata")

    # Stocks
    np.save(os.path.join(output_dir, 'stocks_embeddings.npy'), stock_embeddings)
    stock_df[['finCode', 'name', 'shortName', 'metadata']].to_csv(
        os.path.join(output_dir, 'stocks_metadata.csv'), index=False
    )
    print("Saved stock embeddings and metadata")

    # Holdings
    np.save(os.path.join(output_dir, 'holdings_embeddings.npy'), holdings_embeddings)
    holdings_df[['parentSchemeCode', 'holding_name', 'fund_name', 'metadata']].to_csv(
        os.path.join(output_dir, 'holdings_metadata.csv'), index=False
    )
    print("Saved holdings embeddings and metadata")

    print(f"Total embedding generation time: {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    process_and_save_embeddings()