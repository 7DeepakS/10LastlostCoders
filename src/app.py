# src/app.py
from flask import Flask, request, render_template_string
import numpy as np
import faiss
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from collections import OrderedDict
import time

app = Flask(__name__)

# Load indexes and model once
index_dir = 'data/index/'
mf_index = faiss.read_index(os.path.join(index_dir, 'mutual_funds_index.faiss'))
mf_metadata = pd.read_csv(os.path.join(index_dir, 'mutual_funds_metadata.csv'))
stock_index = faiss.read_index(os.path.join(index_dir, 'stocks_index.faiss'))
stock_metadata = pd.read_csv(os.path.join(index_dir, 'stocks_metadata.csv'))
holdings_index = faiss.read_index(os.path.join(index_dir, 'holdings_index.faiss'))
holdings_metadata = pd.read_csv(os.path.join(index_dir, 'holdings_metadata.csv'))
model = SentenceTransformer('models/finetuned_minilm')  # Assuming latest model
print("Indexes and model loaded")

# Initialize cache (max 100 recent queries)
cache = OrderedDict()
CACHE_SIZE = 100


def search(query, index, metadata, top_k=5):
    """Search the index for top_k matches to the query."""
    start_time = time.time()
    query_embedding = model.encode([query], convert_to_numpy=True)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    distances, indices = index.search(query_embedding, top_k)
    print(f"Full search completed in {time.time() - start_time:.2f} seconds")
    return metadata.iloc[indices[0]], distances[0]


def get_results(query):
    """Get results from cache or perform a new search."""
    query = query.lower().strip()
    if query in cache:
        print(f"Retrieved '{query}' from cache")
        return cache[query], True  # Results, is_cached flag

    # New search
    mf_results, mf_distances = search(query, mf_index, mf_metadata)
    stock_results, stock_distances = search(query, stock_index, stock_metadata)
    holdings_results, holdings_distances = search(query, holdings_index, holdings_metadata)

    results = {
        'mf': (mf_results, mf_distances),
        'stocks': (stock_results, stock_distances),
        'holdings': (holdings_results, holdings_distances)
    }

    # Update cache
    if len(cache) >= CACHE_SIZE:
        cache.popitem(last=False)  # Remove oldest entry
    cache[query] = results
    return results, False


@app.route('/', methods=['GET', 'POST'])
def search_page():
    results_html = ""
    source = ""
    if request.method == 'POST':
        query = request.form['query']
        results, is_cached = get_results(query)
        source = " (Previously Entered)" if is_cached else " (New Search)"

        mf_results, mf_distances = results['mf']
        stock_results, stock_distances = results['stocks']
        holdings_results, holdings_distances = results['holdings']

        results_html = f"""
        <div class="results">
            <h2>Results for: {query}{source}</h2>
            <h3>Mutual Funds</h3>
            <ul>{''.join(f'<li>{row["schemeName"]} - {row["amcName"]}<br><small>{row["metadata"]} (Similarity: {mf_distances[i]:.4f})</small></li>' for i, (_, row) in enumerate(mf_results.iterrows()))}</ul>
            <h3>Stocks</h3>
            <ul>{''.join(f'<li>{row["name"]} ({row["shortName"]})<br><small>{row["metadata"]} (Similarity: {stock_distances[i]:.4f})</small></li>' for i, (_, row) in enumerate(stock_results.iterrows()))}</ul>
            <h3>Holdings</h3>
            <ul>{''.join(f'<li>{row["holding_name"]} in {row["fund_name"]}<br><small>{row["metadata"]} (Similarity: {holdings_distances[i]:.4f})</small></li>' for i, (_, row) in enumerate(holdings_results.iterrows()))}</ul>
        </div>
        """

    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>FindMyFund Search</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; }
            h1 { color: #333; }
            .search-box { margin-bottom: 20px; }
            input[type="text"] { padding: 10px; width: 300px; border: 1px solid #ccc; border-radius: 4px; }
            input[type="submit"] { padding: 10px 20px; background-color: #007BFF; color: white; border: none; border-radius: 4px; cursor: pointer; }
            input[type="submit"]:hover { background-color: #0056b3; }
            .results { background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            h2 { color: #007BFF; }
            h3 { color: #444; margin-top: 20px; }
            ul { list-style-type: none; padding: 0; }
            li { margin: 10px 0; padding: 10px; background-color: #f9f9f9; border-radius: 4px; }
            small { color: #666; font-size: 0.9em; }
            .cache-note { color: #28a745; font-style: italic; }
        </style>
    </head>
    <body>
        <h1>FindMyFund Search</h1>
        <form method="post" class="search-box">
            <input type="text" name="query" placeholder="Enter query (e.g., sbi debt funds)" required>
            <input type="submit" value="Search">
        </form>
        {{ results | safe }}
    </body>
    </html>
    """, results=results_html)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)