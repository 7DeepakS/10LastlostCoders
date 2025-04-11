# src/build_index.py
import numpy as np
import faiss
import pandas as pd
import os
import time


def load_embeddings_and_metadata():
    """Load embeddings and metadata from files."""
    start_time = time.time()
    embeddings_dir = 'data/embeddings/'

    mf_embeddings = np.load(os.path.join(embeddings_dir, 'mutual_funds_embeddings.npy'))
    mf_metadata = pd.read_csv(os.path.join(embeddings_dir, 'mutual_funds_metadata.csv'))

    stock_embeddings = np.load(os.path.join(embeddings_dir, 'stocks_embeddings.npy'))
    stock_metadata = pd.read_csv(os.path.join(embeddings_dir, 'stocks_metadata.csv'))

    holdings_embeddings = np.load(os.path.join(embeddings_dir, 'holdings_embeddings.npy'))
    holdings_metadata = pd.read_csv(os.path.join(embeddings_dir, 'holdings_metadata.csv'))

    # Normalize embeddings for cosine similarity
    mf_embeddings = mf_embeddings / np.linalg.norm(mf_embeddings, axis=1)[:, np.newaxis]
    stock_embeddings = stock_embeddings / np.linalg.norm(stock_embeddings, axis=1)[:, np.newaxis]
    holdings_embeddings = holdings_embeddings / np.linalg.norm(holdings_embeddings, axis=1)[:, np.newaxis]

    print(f"Loaded mutual funds: {len(mf_embeddings)} embeddings")
    print(f"Loaded stocks: {len(stock_embeddings)} embeddings")
    print(f"Loaded holdings: {len(holdings_embeddings)} embeddings")
    print(f"Data loaded in {time.time() - start_time:.2f} seconds")
    return (mf_embeddings, mf_metadata), (stock_embeddings, stock_metadata), (holdings_embeddings, holdings_metadata)


def build_faiss_index(embeddings, use_ivf=False):
    """Build a FAISS index for the given embeddings."""
    start_time = time.time()
    dimension = embeddings.shape[1]  # 384 for MiniLM
    if use_ivf and len(embeddings) > 10000:  # Use IVF for large datasets
        nlist = 100  # Number of clusters
        quantizer = faiss.IndexFlatIP(dimension)  # Inner Product for cosine
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.nprobe = 10  # Trade-off between speed and accuracy
    else:
        index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine
    index.add(embeddings)
    print(f"Built FAISS index with {index.ntotal} vectors in {time.time() - start_time:.2f} seconds")
    return index


def build_and_save_indexes():
    """Build and save FAISS indexes for all embeddings."""
    start_time = time.time()
    (mf_embeddings, mf_metadata), (stock_embeddings, stock_metadata), (
    holdings_embeddings, holdings_metadata) = load_embeddings_and_metadata()

    # Build indexes
    print("Building mutual funds index...")
    mf_index = build_faiss_index(mf_embeddings)
    print("Building stocks index...")
    stock_index = build_faiss_index(stock_embeddings)
    print("Building holdings index...")
    holdings_index = build_faiss_index(holdings_embeddings, use_ivf=True)

    # Save indexes
    index_dir = 'data/index/'
    os.makedirs(index_dir, exist_ok=True)

    faiss.write_index(mf_index, os.path.join(index_dir, 'mutual_funds_index.faiss'))
    mf_metadata.to_csv(os.path.join(index_dir, 'mutual_funds_metadata.csv'), index=False)
    print("Saved mutual funds index and metadata")

    faiss.write_index(stock_index, os.path.join(index_dir, 'stocks_index.faiss'))
    stock_metadata.to_csv(os.path.join(index_dir, 'stocks_metadata.csv'), index=False)
    print("Saved stocks index and metadata")

    faiss.write_index(holdings_index, os.path.join(index_dir, 'holdings_index.faiss'))
    holdings_metadata.to_csv(os.path.join(index_dir, 'holdings_metadata.csv'), index=False)
    print("Saved holdings index and metadata")

    print(f"Total index building time: {time.time() - start_time:.2f} seconds")


if __name__ == '__main__':
    build_and_save_indexes()