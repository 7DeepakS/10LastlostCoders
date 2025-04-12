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
model = SentenceTransformer('models/finetuned_minilm_v4')
print("Indexes and model loaded")

# Initialize cache (max 100 recent queries)
cache = OrderedDict()
CACHE_SIZE = 100


def search(query, index, metadata, top_k=1):
    """Search the index for top_k matches to the query."""
    start_time = time.time()
    query_embedding = model.encode([query], convert_to_numpy=True)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    distances, indices = index.search(query_embedding, top_k)
    results = metadata.iloc[indices[0]].drop_duplicates()
    print(f"Search indices: {indices[0]}, Distances: {distances[0]}")
    print(f"Search completed in {time.time() - start_time:.2f} seconds")
    return results, distances[0]


def get_results(query):
    """Get results from cache or perform a new search."""
    query = query.lower().strip()
    if query in cache:
        print(f"Retrieved '{query}' from cache")
        return cache[query], True

    mf_results, mf_distances = search(query, mf_index, mf_metadata)
    stock_results, stock_distances = search(query, stock_index, stock_metadata)
    holdings_results, holdings_distances = search(query, holdings_index, holdings_metadata)

    results = {
        'mf': (mf_results, mf_distances),
        'stocks': (stock_results, stock_distances),
        'holdings': (holdings_results, holdings_distances)
    }

    if len(cache) >= CACHE_SIZE:
        cache.popitem(last=False)
    cache[query] = results
    return results, False


@app.route('/', methods=['GET', 'POST'])
def search_page():
    results_html = ""
    if request.method == 'POST':
        query = request.form['query']
        if 'clear_cache' in request.form:
            cache.clear()
            print("Cache cleared")
            results_html = "<p class='cache-note'>Cache cleared!</p>"
        else:
            results, is_cached = get_results(query)
            source = " (From Cache)" if is_cached else " (New Search)"

            mf_results, mf_distances = results['mf']
            stock_results, stock_distances = results['stocks']
            holdings_results, holdings_distances = results['holdings']

            results_html = f"""
            <div class="results-container">
                <h2>Results for: {query}{source}</h2>
                <div class="category">
                    <h3>Mutual Funds ({len(mf_results)} found)</h3>
                    <div class="cards">
                        {''.join(f'<div class="card"><h4>{row["schemeName"]}</h4><p>{row["amcName"]}</p><p class="metadata">{row["metadata"]}</p><p class="similarity">Similarity: {mf_distances[i]:.4f}</p></div>' for i, (_, row) in enumerate(mf_results.iterrows()))}
                    </div>
                </div>
                <div class="category">
                    <h3>Stocks ({len(stock_results)} found)</h3>
                    <div class="cards">
                        {''.join(f'<div class="card"><h4>{row["name"]} ({row["shortName"]})</h4><p class="metadata">{row["metadata"]}</p><p class="similarity">Similarity: {stock_distances[i]:.4f}</p></div>' for i, (_, row) in enumerate(stock_results.iterrows()))}
                    </div>
                </div>
                <div class="category">
                    <h3>Holdings ({len(holdings_results)} found)</h3>
                    <div class="cards">
                        {''.join(f'<div class="card"><h4>{row["holding_name"]}</h4><p>In: {row["fund_name"]}</p><p class="metadata">{row["metadata"]}</p><p class="similarity">Similarity: {holdings_distances[i]:.4f}</p></div>' for i, (_, row) in enumerate(holdings_results.iterrows()))}
                    </div>
                </div>
            </div>
            """

    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>FindMyFund Search</title>
        <style>
            body {
                font-family: 'Segoe UI', Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f0f2f5;
                color: #333;
            }
            h1 {
                text-align: center;
                color: #1a73e8;
                margin-bottom: 30px;
            }
            .search-box {
                display: flex;
                justify-content: center;
                gap: 10px;
                margin-bottom: 30px;
            }
            input[type="text"] {
                padding: 12px;
                width: 400px;
                border: 2px solid #ddd;
                border-radius: 25px;
                font-size: 16px;
                outline: none;
                transition: border-color 0.3s;
            }
            input[type="text"]:focus {
                border-color: #1a73e8;
            }
            input[type="submit"] {
                padding: 12px 25px;
                background-color: #1a73e8;
                color: white;
                border: none;
                border-radius: 25px;
                font-size: 16px;
                cursor: pointer;
                transition: background-color 0.3s;
            }
            input[type="submit"]:hover {
                background-color: #1557b0;
            }
            .clear-cache {
                padding: 12px 25px;
                background-color: #e91e63;
                color: white;
                border: none;
                border-radius: 25px;
                font-size: 16px;
                cursor: pointer;
                transition: background-color 0.3s;
            }
            .clear-cache:hover {
                background-color: #c2185b;
            }
            .results-container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                animation: fadeIn 0.5s;
            }
            h2 {
                color: #1a73e8;
                text-align: center;
                margin-bottom: 20px;
            }
            .category {
                margin-bottom: 30px;
            }
            h3 {
                color: #444;
                border-bottom: 2px solid #1a73e8;
                padding-bottom: 5px;
            }
            .cards {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
            }
            .card {
                background-color: #fff;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
                transition: transform 0.2s, box-shadow 0.2s;
            }
            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
            }
            .card h4 {
                margin: 0 0 10px;
                color: #1a73e8;
                font-size: 18px;
            }
            .card p {
                margin: 5px 0;
                font-size: 14px;
            }
            .metadata {
                color: #666;
                font-style: italic;
            }
            .similarity {
                color: #28a745;
                font-weight: bold;
            }
            .cache-note {
                text-align: center;
                color: #28a745;
                font-style: italic;
            }
            #loading {
                display: none;
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                font-size: 20px;
                color: #1a73e8;
            }
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            @media (max-width: 600px) {
                .search-box { flex-direction: column; align-items: center; }
                input[type="text"] { width: 100%; margin-bottom: 10px; }
            }
        </style>
        <script>
            function showLoading() {
                document.getElementById('loading').style.display = 'block';
            }
        </script>
    </head>
    <body>
        <h1>FindMyFund Search</h1>
        <form method="post" class="search-box" onsubmit="showLoading()">
            <input type="text" name="query" placeholder="Enter query (e.g., sbi debt funds)" required>
            <input type="submit" value="Search">
            <input type="submit" name="clear_cache" value="Clear Cache" class="clear-cache">
        </form>
        <div id="loading">Searching...</div>
        {{ results | safe }}
    </body>
    </html>
    """, results=results_html)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)