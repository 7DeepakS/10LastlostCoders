# src/preprocess.py
import pandas as pd
import json
import os
import time

def load_json(file_path):
    """Load a JSON file and return its contents."""
    start_time = time.time()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {file_path} in {time.time() - start_time:.2f} seconds")
        return data
    except json.JSONDecodeError as e:
        print(f"JSON error in {file_path}: {e}")
        return []
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def normalize_text_series(series):
    """Vectorized text normalization without spell checking."""
    return series.fillna('').str.lower()

def preprocess_data():
    """Preprocess mutual funds, stocks, and holdings data efficiently."""
    start_time = time.time()
    # Define input and output paths
    input_dir = 'data/raw/'
    output_dir = 'data/processed/'
    os.makedirs(output_dir, exist_ok=True)

    # Load JSON files
    mf_data = load_json(os.path.join(input_dir, 'mutual_funds_data.json'))
    stock_data = load_json(os.path.join(input_dir, 'stock_data.json'))
    holdings_data = load_json(os.path.join(input_dir, 'mf_holdings_data.json'))

    # Convert to DataFrames
    mf_df = pd.DataFrame(mf_data)
    stock_df = pd.DataFrame(stock_data)
    holdings_df = pd.DataFrame(holdings_data)

    # Debug: Print sample data
    print("Mutual Funds sample:", mf_df[['schemeName', 'amcName']].head(2).to_dict())
    print("Stocks sample:", stock_df[['name', 'shortName']].head(2).to_dict())
    print("Holdings sample:", holdings_df[['name', 'parentSchemeCode']].head(2).to_dict())

    # Process mutual funds
    if not mf_df.empty and 'schemeName' in mf_df.columns:
        mf_df = mf_df.dropna(subset=['schemeName'])
        mf_df['schemeName'] = normalize_text_series(mf_df['schemeName'])
        mf_df['amcName'] = normalize_text_series(mf_df['amcName'])
        mf_df['category'] = normalize_text_series(mf_df['category'])
        mf_df['subCategory'] = normalize_text_series(mf_df['subCategory'])
        if 'shortSchemeDescription' in mf_df:
            mf_df['shortSchemeDescription'] = normalize_text_series(mf_df['shortSchemeDescription'])

        # Create metadata
        mf_df['metadata'] = (
            "[Fund House: " + mf_df['amcName'] + ", Category: " + mf_df['category'] +
            ", Sub-Category: " + mf_df['subCategory'] + ", AUM: " + mf_df['aum'].fillna('N/A').astype(str) + "]"
        )
    else:
        print("Mutual funds DataFrame is empty or missing 'schemeName' column")

    # Process stocks
    if not stock_df.empty and 'name' in stock_df.columns:
        stock_df = stock_df.dropna(subset=['name'])
        stock_df['name'] = normalize_text_series(stock_df['name'])
        stock_df['shortName'] = normalize_text_series(stock_df['shortName'])
        stock_df['sector'] = normalize_text_series(stock_df['sector'])
        stock_df['industry'] = normalize_text_series(stock_df['industry'])
        stock_df['category'] = normalize_text_series(stock_df['category'])

        # Create metadata
        stock_df['metadata'] = (
            "[Sector: " + stock_df['sector'] + ", Industry: " + stock_df['industry'] +
            ", Category: " + stock_df['category'] + ", Security Type: " + stock_df['securityType'] + "]"
        )
    else:
        print("Stocks DataFrame is empty or missing 'name' column")

    # Process holdings
    if not holdings_df.empty and 'name' in holdings_df.columns:
        holdings_df['name'] = holdings_df['name'].fillna('unknown holding')
        holdings_df['name'] = normalize_text_series(holdings_df['name'])
        holdings_df['shortName'] = normalize_text_series(holdings_df['shortName'].fillna('unknown'))
        holdings_df['sector'] = normalize_text_series(holdings_df['sector'])
        holdings_df['industry'] = normalize_text_series(holdings_df['industry'])
        holdings_df['category'] = normalize_text_series(holdings_df['category'])

        # Create metadata
        holdings_df['metadata'] = (
            "[Asset Type: " + holdings_df['assetType'] + ", Sector: " + holdings_df['sector'] +
            ", Industry: " + holdings_df['industry'] + ", Category: " + holdings_df['category'] + "]"
        )

        # Optimize merge with indexing
        if not mf_df.empty and 'schemeCode' in mf_df.columns:
            mf_df.set_index('schemeCode', inplace=True)
            holdings_df = holdings_df.merge(
                mf_df[['schemeName']],
                left_on='parentSchemeCode',
                right_index=True,
                how='left'
            )
            mf_df.reset_index(inplace=True)
            holdings_df['fund_name'] = holdings_df['schemeName'].fillna('unknown')
            holdings_df['holding_name'] = holdings_df['name']
    else:
        print("Holdings DataFrame is empty or missing 'name' column")

    # Generate synthetic queries (reduced set for speed)
    synthetic_queries = []

    if not mf_df.empty and 'schemeName' in mf_df.columns:
        for _, row in mf_df.iterrows():
            name = row['schemeName']
            if name:
                synthetic_queries.append({'query': name.split()[0], 'target': name, 'type': 'mutual_fund'})
                synthetic_queries.append({'query': f"{row['amcName'].split()[0]} funds", 'target': name, 'type': 'mutual_fund'})

    if not stock_df.empty and 'name' in stock_df.columns:
        for _, row in stock_df.iterrows():
            name = row['name']
            if name:
                synthetic_queries.append({'query': name, 'target': name, 'type': 'stock'})
                synthetic_queries.append({'query': f"{row['sector']} stocks", 'target': name, 'type': 'stock'})

    if not holdings_df.empty and 'holding_name' in holdings_df.columns:
        for _, row in holdings_df.iterrows():
            fund_name = row['fund_name']
            holding_name = row['holding_name']
            if fund_name != 'unknown' and holding_name != 'unknown holding':
                synthetic_queries.append({'query': f"funds with {holding_name} holdings", 'target': fund_name, 'type': 'mutual_fund'})

    query_df = pd.DataFrame(synthetic_queries)

    # Save processed data
    mf_df.to_csv(os.path.join(output_dir, 'mutual_funds.csv'), index=False)
    stock_df.to_csv(os.path.join(output_dir, 'stocks.csv'), index=False)
    holdings_df.to_csv(os.path.join(output_dir, 'holdings.csv'), index=False)
    query_df.to_csv(os.path.join(output_dir, 'queries.csv'), index=False)

    print(f"Processed data saved to {output_dir}")
    print(f"Mutual Funds: {len(mf_df)} records")
    print(f"Stocks: {len(stock_df)} records")
    print(f"Holdings: {len(holdings_df)} records")
    print(f"Synthetic Queries: {len(query_df)} records")
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    preprocess_data()