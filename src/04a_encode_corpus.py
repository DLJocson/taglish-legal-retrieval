"""Encode the passage corpus with multiple embedding models and build FAISS indices.

Loads processed passages, persists a row-to-passage ID mapping, encodes with
mSBERT, BGE-M3, and Legal-BERT, and saves normalized embeddings and
IndexFlatIP FAISS indices under ``data/processed/indices/``.
"""

import json
import os
import time

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

print("========================================")
print("   MODULE 2: CORPUS ENCODING PIPELINE   ")
print("========================================")

print("Loading processed passages...")
df = pd.read_csv("data/processed/module1_processed_passages.csv")

passages = df['passage_text'].astype(str).tolist()
passage_ids = df['passage_id'].tolist()

print(f"Loaded {len(passages)} passages.")

os.makedirs("data/processed/indices", exist_ok=True)

mapping_path = "data/processed/indices/id_mapping.json"
with open(mapping_path, 'w') as f:
    json.dump(passage_ids, f)
print(f"Saved FAISS row-to-ID mapping at {mapping_path}")

models_to_run = {
    "msbert": "paraphrase-multilingual-mpnet-base-v2",
    "bge_m3": "BAAI/bge-m3",
    "legal_bert": "nlpaueb/legal-bert-base-uncased"
}

for model_alias, model_path in models_to_run.items():
    print(f"\n----------------------------------------")
    print(f"Processing Model: {model_alias.upper()}")
    print(f"Source: {model_path}")
    print(f"----------------------------------------")

    try:
        print("Loading model weights into memory...")
        print("(This will download the model to your cache if it is the first time)")
        model = SentenceTransformer(model_path)

        # BGE-M3 is memory-heavy; smaller batches reduce OOM risk.
        current_batch_size = 16 if "bge-m3" in model_path.lower() else 64

        print(f"Encoding passages with batch_size={current_batch_size}... (This will take time!)")
        start_time = time.time()

        embeddings = model.encode(
            passages,
            batch_size=current_batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        encoding_time = time.time() - start_time
        print(f"Encoding completed in {encoding_time:.2f} seconds.")

        print("Building FAISS IndexFlatIP (Cosine Similarity)...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        npy_path = f"data/processed/indices/{model_alias}_embeddings.npy"
        faiss_path = f"data/processed/indices/{model_alias}_index.faiss"

        np.save(npy_path, embeddings)
        faiss.write_index(index, faiss_path)

        print(f"Successfully saved {model_alias} index with {index.ntotal} vectors.")

    except Exception as e:
        print(f"FAILED to process {model_alias}. Error: {e}")
        print("Moving to the next model...")

print("\n========================================")
print("   MODULE 2 COMPLETE! All models encoded. ")
print("========================================")
