# src/finetune.py
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import os
import time

def load_data():
    """Load preprocessed queries."""
    start_time = time.time()
    queries_df = pd.read_csv('data/processed/que.csv')
    print(f"Loaded queries: {len(queries_df)} records in {time.time() - start_time:.2f} seconds")
    return queries_df

def prepare_training_data(queries_df):
    """Prepare training examples from query-target pairs."""
    start_time = time.time()
    train_examples = [
        InputExample(texts=[row['query'], row['target']])
        for _, row in queries_df.iterrows()
    ]
    print(f"Prepared {len(train_examples)} training examples in {time.time() - start_time:.2f} seconds")
    return train_examples

def finetune_model():
    """Fine-tune MiniLM model with query-target pairs."""
    start_time = time.time()
    # Load pre-trained model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded")

    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = model.to(device)

    # Load and prepare data
    queries_df = load_data()
    train_examples = prepare_training_data(queries_df)

    # Define DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
    train_loss = losses.MultipleNegativesRankingLoss(model)  # Optimized for ranking

    # Fine-tune
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=200,  # Increased for better accuracy
        warmup_steps=100,
        output_path='models/finetuned_minilm_v4',
        show_progress_bar=True
    )
    print(f"Model fine-tuned and saved to models/finetuned_minilm_v4 in {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    finetune_model()