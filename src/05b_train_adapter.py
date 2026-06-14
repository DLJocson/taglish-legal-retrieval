import ast
import json
import os
import time

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer


script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
BASE_PATH = os.path.join(project_root, "data", "processed")
INDICES_DIR = os.path.join(BASE_PATH, "indices")
QUERIES_FILE = os.path.join(BASE_PATH, "queries_annotated.csv")
ID_MAPPING_FILE = os.path.join(INDICES_DIR, "id_mapping.json")

MODELS_TO_TRAIN = {
    "msbert": "paraphrase-multilingual-mpnet-base-v2",
    "bge_m3": "BAAI/bge-m3",
    "legal_bert": "nlpaueb/legal-bert-base-uncased",
}

# Training configuration
ALLOWED_LANGUAGES = {"English", "Tagalog", "Code-Switched"}
TRIPLET_MARGIN = 0.3  # Margin for triplet loss: larger margin forces better separation
LEARNING_RATE = 2e-4  # Conservative LR to avoid overfitting on small dataset
EPOCHS = 50  # Sufficient epochs for convergence with early stopping via best checkpoint
TOP_K = 10  # Retrieve top-10 for hard negative mining
MIN_TRIPLETS = 10  # Minimum triplets required to attempt training
WEIGHT_DECAY = 1e-3  # L2 regularization to prevent overfitting

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_list_value(value):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    return [str(v).strip() for v in parsed if str(v).strip()]
            except Exception:
                try:
                    parsed = json.loads(text.replace("'", '"'))
                    if isinstance(parsed, list):
                        return [str(v).strip() for v in parsed if str(v).strip()]
                except Exception:
                    return []
    return []


def normalize_rows(x):
    x = x.astype(np.float32, copy=False)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms


def load_passage_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    if isinstance(mapping, list):
        return [str(x) for x in mapping]

    if isinstance(mapping, dict):
        if all(str(k).isdigit() for k in mapping.keys()):
            ordered = [""] * len(mapping)
            for k, v in mapping.items():
                ordered[int(k)] = str(v)
            return ordered

        if all(str(v).isdigit() for v in mapping.values()):
            ordered = [""] * len(mapping)
            for k, v in mapping.items():
                ordered[int(v)] = str(k)
            return ordered

    raise ValueError("Unsupported id_mapping.json format.")


class LinearAdapter(nn.Module):
    # Neural adapter for aligning query and passage embeddings (LINAW)
    # Initialized as identity transformation (eye_) to start close to baseline
    # LayerNorm + ReLU provides non-linearity while preserving dimensionality
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=True)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.ReLU()
        nn.init.eye_(self.proj.weight)  # Initialize as identity for stable training start
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        return self.act(self.norm(self.proj(x)))


def build_triplets(queries_df, passage_index, passage_ids, passage_embeddings, query_encoder, top_k=10):
    index = faiss.IndexFlatIP(passage_embeddings.shape[1])
    index.add(passage_embeddings)

    triplets = []
    skipped_no_positive = 0
    skipped_no_negative = 0

    for _, row in queries_df.iterrows():
        relevant_ids = set(parse_list_value(row["relevant_passage_ids"]))
        if not relevant_ids:
            continue

        query_vec = query_encoder.encode(
            [str(row["query_text"])],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        _, faiss_indices = index.search(query_vec, top_k)

        retrieved_ids = []
        for fi in faiss_indices[0]:
            if fi != -1:
                retrieved_ids.append(str(passage_ids[fi]))

        positives = [pid for pid in retrieved_ids if pid in relevant_ids]
        negatives = [pid for pid in retrieved_ids if pid not in relevant_ids]

        if not positives:
            skipped_no_positive += 1
            continue
        if not negatives:
            skipped_no_negative += 1
            continue

        q_vec = query_vec[0]
        
        # Generate all positive-negative combinations for comprehensive training signal
        # This creates more triplets than random sampling, improving adapter convergence
        for pos_id in positives:
            pos_vec = passage_embeddings[passage_index[pos_id]]
            for neg_id in negatives:
                neg_vec = passage_embeddings[passage_index[neg_id]]
                triplets.append((q_vec, pos_vec, neg_vec))

    print(f"  Triplets formed       : {len(triplets)}")
    print(f"  Skipped (no positive) : {skipped_no_positive}")
    print(f"  Skipped (no negative) : {skipped_no_negative}")
    return triplets


def train_adapter(triplets, dim, alias):
    adapter = LinearAdapter(dim).to(DEVICE)
    
    # Use cosine distance (1 - cosine_similarity) as triplet loss distance metric
    # This matches the retrieval similarity metric (IndexFlatIP uses inner product on normalized vectors)
    # Ensures adapter optimizes for the same similarity used during inference
    cosine_distance = lambda x, y: 1.0 - nn.functional.cosine_similarity(x, y, dim=-1)
    criterion = nn.TripletMarginWithDistanceLoss(
        distance_function=cosine_distance, 
        margin=TRIPLET_MARGIN, 
        reduction="mean"
    )
    
    optimizer = optim.Adam(adapter.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    q_tensors = torch.from_numpy(np.stack([t[0] for t in triplets]).astype(np.float32)).to(DEVICE)
    pos_tensors = torch.from_numpy(np.stack([t[1] for t in triplets]).astype(np.float32)).to(DEVICE)
    neg_tensors = torch.from_numpy(np.stack([t[2] for t in triplets]).astype(np.float32)).to(DEVICE)

    pos_tensors = nn.functional.normalize(pos_tensors, p=2, dim=-1)
    neg_tensors = nn.functional.normalize(neg_tensors, p=2, dim=-1)

    print(f"\n  Training adapter for {alias.upper()} ({len(triplets)} triplets, {EPOCHS} epochs)...")

    best_loss = float("inf")
    best_state = None
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        adapter.train()
        optimizer.zero_grad()

        q_proj = nn.functional.normalize(adapter(q_tensors), p=2, dim=-1)
        loss = criterion(q_proj, pos_tensors, neg_tensors)

        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}

        if epoch == 1 or epoch % 5 == 0:
            elapsed = time.time() - start
            print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={loss_val:.6f}  best={best_loss:.6f}  elapsed={elapsed:.1f}s")

    if best_state is not None:
        adapter.load_state_dict(best_state)
        print(f"  Restored best checkpoint (loss={best_loss:.6f})")

    return adapter


def main():
    print("========================================")
    print("  MODULE 2.5: NEURAL ADAPTER TRAINING   ")
    print("========================================")
    print(f"Using device: {DEVICE}")

    os.makedirs(INDICES_DIR, exist_ok=True)

    if not os.path.exists(QUERIES_FILE):
        raise FileNotFoundError(f"Missing annotated queries file: {QUERIES_FILE}")
    if not os.path.exists(ID_MAPPING_FILE):
        raise FileNotFoundError(f"Missing passage mapping file: {ID_MAPPING_FILE}")

    print("\nLoading annotated queries...")
    queries_df = pd.read_csv(QUERIES_FILE)
    queries_df["language_label"] = queries_df["language_label"].astype(str).str.strip()
    queries_df["relevant_passage_ids"] = queries_df["relevant_passage_ids"].apply(parse_list_value)

    # Filter to training split only to prevent data leakage
    # Training on test queries would invalidate evaluation metrics
    if "split" in queries_df.columns:
        in_scope_df = queries_df[
            queries_df["language_label"].isin(ALLOWED_LANGUAGES)
            & queries_df["relevant_passage_ids"].map(len).gt(0)
            & (queries_df["split"] == "train")  # Filters exclusively for the training set
        ].copy()
    else:
        print("\n!!! CRITICAL WARNING: 'split' column missing in queries_annotated.csv. !!!")
        print("!!! Training on ALL data. This causes data leakage and invalidates evaluation. !!!")
        print("!!! Please add a 'split' column containing 'train' and 'test' values. !!!\n")
        in_scope_df = queries_df[
            queries_df["language_label"].isin(ALLOWED_LANGUAGES)
            & queries_df["relevant_passage_ids"].map(len).gt(0)
        ].copy()

    print(f"In-scope annotated queries available for training: {len(in_scope_df)}")
    if len(in_scope_df) < MIN_TRIPLETS:
        raise ValueError(f"Only {len(in_scope_df)} usable queries found. Need at least {MIN_TRIPLETS}.")

    passage_ids = load_passage_ids(ID_MAPPING_FILE)
    passage_index = {pid: i for i, pid in enumerate(passage_ids)}

    for alias, model_path in MODELS_TO_TRAIN.items():
        print(f"\n{'=' * 50}")
        print(f"  MODEL: {alias.upper()}")
        print(f"{'=' * 50}")

        adapter_save_path = os.path.join(INDICES_DIR, f"{alias}_adapter.pt")
        if os.path.exists(adapter_save_path):
            print(f"  Adapter already exists at {adapter_save_path}. Skipping.")
            continue

        npy_path = os.path.join(INDICES_DIR, f"{alias}_embeddings.npy")
        if not os.path.exists(npy_path):
            print(f"  WARNING: {npy_path} not found. Skipping {alias}.")
            continue

        print(f"  Loading raw embeddings from {npy_path}...")
        raw_embeddings = np.load(npy_path).astype(np.float32)
        passage_embeddings = normalize_rows(raw_embeddings)
        dim = passage_embeddings.shape[1]
        print(f"  Embedding matrix shape : {passage_embeddings.shape}")

        print(f"  Loading sentence encoder: {model_path}")
        encoder = SentenceTransformer(model_path, device="cpu")

        # Mine hard negatives from top-K retrieved passages
        # Hard negatives are passages retrieved by the model but not marked as relevant
        # Training on these improves the model's ability to distinguish similar but incorrect passages
        print(f"  Mining hard negatives from top-{TOP_K} retrieved passages...")
        triplets = build_triplets(
            queries_df=in_scope_df,
            passage_index=passage_index,
            passage_ids=passage_ids,
            passage_embeddings=passage_embeddings,
            query_encoder=encoder,
            top_k=TOP_K,
        )

        if len(triplets) < MIN_TRIPLETS:
            print(f"  Only {len(triplets)} triplets formed. Skipping training for {alias}.")
            continue

        adapter = train_adapter(triplets, dim, alias)

        torch.save(
            {
                "state_dict": adapter.state_dict(),
                "alias": alias,
                "model_path": model_path,
                "embedding_dim": dim,
                "triplet_margin": TRIPLET_MARGIN,
                "learning_rate": LEARNING_RATE,
                "epochs": EPOCHS,
            },
            adapter_save_path,
        )

        print(f"  Saved adapter weights: {adapter_save_path}")

        del raw_embeddings, passage_embeddings, encoder, adapter, triplets
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n========================================")
    print("  MODULE 2.5 COMPLETE                   ")
    print("  Next: apply adapter to queries only   ")
    print("  (passages remain unchanged; adapter shifts query space)")
    print("========================================")


if __name__ == "__main__":
    main()