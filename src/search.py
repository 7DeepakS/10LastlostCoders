# src/search.py
import numpy as np
import os
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import time


def load_indexes_and_model():
    """Load FAISS indexes, metadata, and fine-tuned model."""
    start_time = time.time()
    index_dir = 'data/index/'

    mf_index = faiss.read_index(os.path.join(index_dir, 'mutual_funds_index.faiss'))
    mf_metadata = pd.read_csv(os.path.join(index_dir, 'mutual_funds_metadata.csv'))

    stock_index = faiss.read_index(os.path.join(index_dir, 'stocks_index.faiss'))
    stock_metadata = pd.read_csv(os.path.join(index_dir, 'stocks_metadata.csv'))

    holdings_index = faiss.read_index(os.path.join(index_dir, 'holdings_index.faiss'))
    holdings_metadata = pd.read_csv(os.path.join(index_dir, 'holdings_metadata.csv'))

    model = SentenceTransformer('models/finetuned_minilm')
    print(f"Indexes and model loaded in {time.time() - start_time:.2f} seconds")
    return (mf_index, mf_metadata), (stock_index, stock_metadata), (holdings_index, holdings_metadata), model


def search(query, model, index, metadata, top_k=5):
    """Search the index for top_k matches to the query."""
    start_time = time.time()
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    results = metadata.iloc[indices[0]]
    print(f"Search completed in {time.time() - start_time:.2f} seconds")
    return results, distances[0]


def run_interactive_search():
    """Run an interactive search CLI."""
    (mf_index, mf_metadata), (stock_index, stock_metadata), (
    holdings_index, holdings_metadata), model = load_indexes_and_model()

    print("\nWelcome to FindMyFund Search!")
    print("Enter a query (e.g., 'sbi debt funds', 'materials stocks') or 'exit' to quit.")

    while True:
        query = input("\nQuery: ").strip()
        if query.lower() == 'exit':
            print("Goodbye!")
            break

        if not query:
            print("Please enter a valid query.")
            continue

        print(f"\nSearching for: {query}")

        print("\nTop 5 Mutual Funds:")
        mf_results, mf_distances = search(query, model, mf_index, mf_metadata)
        for i, (idx, row) in enumerate(mf_results.iterrows()):
            print(f"{i + 1}. {row['schemeName']} (Distance: {mf_distances[i]:.4f})")

        print("\nTop 5 Stocks:")
        stock_results, stock_distances = search(query, model, stock_index, stock_metadata)
        for i, (idx, row) in enumerate(stock_results.iterrows()):
            print(f"{i + 1}. {row['name']} (Distance: {stock_distances[i]:.4f})")

        print("\nTop 5 Holdings:")
        holdings_results, holdings_distances = search(query, model, holdings_index, holdings_metadata)
        for i, (idx, row) in enumerate(holdings_results.iterrows()):
            print(f"{i + 1}. {row['holding_name']} in {row['fund_name']} (Distance: {holdings_distances[i]:.4f})")


if __name__ == '__main__':
    run_interactive_search()