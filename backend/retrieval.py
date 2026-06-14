"""FAISS retrieval, corpus loading, and passage metadata."""

from __future__ import annotations

import json
import os
from typing import Any

import faiss
import numpy as np
import pandas as pd
from langdetect import detect_langs
from sentence_transformers import SentenceTransformer

from backend.config import (
    INDICES_DIR,
    MASTER_CORPUS_CSV,
    METRICS_DETAILED_ALIGNED_CSV,
    METRICS_DETAILED_BASELINE_CSV,
    MODEL_CONFIG,
    PASSAGES_CSV,
)
from backend.display_title import load_doc_metadata_by_url, resolve_display_title


# Global CSV data loading for demo metrics (loaded once at module import)
try:
    BASELINE_METRICS_DF = pd.read_csv(METRICS_DETAILED_BASELINE_CSV)
    ALIGNED_METRICS_DF = pd.read_csv(METRICS_DETAILED_ALIGNED_CSV)
except FileNotFoundError:
    BASELINE_METRICS_DF = pd.DataFrame()
    ALIGNED_METRICS_DF = pd.DataFrame()


# Mapping of exact query texts to their query IDs for demo metrics lookup
BENCHMARK_QUERIES: dict[str, str] = {
    "Pwede bang idemanda ang ex na naglalabas ng private chat screenshots?": "Q_013",
    "Pwede bang magreklamo kung hindi binibigay ng employer ang 13th month pay?": "Q_007",
}


def get_demo_metrics(query_text: str, model_name: str) -> dict[str, Any] | None:
    """Retrieve demo metrics for benchmark queries.

    Args:
        query_text: The user's query text.
        model_name: The model key from MODEL_CONFIG (e.g., "BGE-M3", "mSBERT", "Legal-BERT").

    Returns:
        Dictionary with metrics structure: { "mrr": { "baseline": 0.0, "aligned": 0.25, "improvement": 0.25 }, ... }
        or None if the query is not in the benchmark set.
    """
    if BASELINE_METRICS_DF.empty or ALIGNED_METRICS_DF.empty:
        return None

    # Look up query ID from benchmark mapping
    query_id = BENCHMARK_QUERIES.get(query_text)
    if not query_id:
        return None

    # Map model name to CSV column naming convention
    model_csv_name = {
        "BGE-M3": "BGE_M3",
        "mSBERT": "MSBERT",
        "Legal-BERT": "LEGAL_BERT",
    }.get(model_name)

    if not model_csv_name:
        return None

    # Query the DataFrames for the specific query and model
    baseline_row = BASELINE_METRICS_DF[
        (BASELINE_METRICS_DF["Query_ID"] == query_id) & 
        (BASELINE_METRICS_DF["Model"] == model_csv_name)
    ]
    aligned_row = ALIGNED_METRICS_DF[
        (ALIGNED_METRICS_DF["Query_ID"] == query_id) & 
        (ALIGNED_METRICS_DF["Model"] == model_csv_name)
    ]

    if baseline_row.empty or aligned_row.empty:
        return None

    baseline = baseline_row.iloc[0]
    aligned = aligned_row.iloc[0]

    # Extract metric values and calculate improvements
    metrics = {}
    for metric in ["MRR", "P@5", "P@10", "Recall@10"]:
        baseline_val = float(baseline.get(metric, 0.0))
        aligned_val = float(aligned.get(metric, 0.0))
        improvement = aligned_val - baseline_val

        metric_key = metric.lower().replace("@", "_").replace("recall@", "recall_")
        metrics[metric_key] = {
            "baseline": round(baseline_val, 2),
            "aligned": round(aligned_val, 2),
            "improvement": round(improvement, 2),
        }

    return metrics


class AppState:
    # Global singleton holding loaded models, indices, and corpus data
    # Initialized during FastAPI lifespan to avoid reloading on each request
    passages_df: pd.DataFrame
    id_mapping: list[str]
    faiss_indices: dict[str, faiss.Index]
    models: dict[str, SentenceTransformer]
    lang_col: str | None
    type_col: str | None
    doc_metadata_by_url: dict[str, dict[str, str | None]]


state = AppState()
state.faiss_indices = {}
state.models = {}
state.doc_metadata_by_url = {}
state.adapters = {}  # Neural adapters for LINAW alignment


def identify_language(text: str) -> str:
    """Detect query language using dual-threshold heuristic for code-switching.

    Args:
        text: Raw query string.

    Returns:
        "Code-Switched", "Tagalog", "English", or "Other"; defaults to "English".
    """
    try:
        predictions = detect_langs(text)
        res = {l.lang: l.prob for l in predictions}

        # Dual-threshold: both English and Tagalog must exceed 20% to qualify as code-switched
        # This avoids false positives from loanwords or minor code-mixing
        if 'en' in res and 'tl' in res and min(res['en'], res['tl']) > 0.20:
            return "Code-Switched"

        dominant = predictions[0].lang
        return "Tagalog" if dominant == 'tl' else "English" if dominant == 'en' else "Other"
    except:
        # Fallback to English on detection failure (e.g., very short queries)
        return "English"


def cell_str(value: Any) -> str | None:
    # Safely convert DataFrame cell values to strings, handling NaN/None/"NaN" strings
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    return s if s and s.lower() not in ("none", "nan") else None


def passage_titles(r: pd.Series) -> tuple[str | None, str | None]:
    short_title = cell_str(r.get("short_title"))
    document_title = cell_str(r.get("document_title"))
    source_url = cell_str(r.get("source_url"))
    if source_url:
        url_meta = state.doc_metadata_by_url.get(source_url, {})
        if not short_title:
            short_title = cell_str(url_meta.get("short_title"))
        if not document_title:
            document_title = cell_str(url_meta.get("document_title"))
    return short_title, document_title


def load_passages() -> pd.DataFrame:
    if not PASSAGES_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(PASSAGES_CSV)
    if "passage_id" in df.columns:
        df["passage_id"] = df["passage_id"].astype(str)
    return df


def load_id_mapping() -> list[str]:
    path = INDICES_DIR / "id_mapping.json"
    if not path.is_file():
        raise FileNotFoundError(f"ID mapping not found: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_faiss_index(path: str) -> faiss.Index:
    if not os.path.exists(path):
        raise FileNotFoundError(f"FAISS index not found: {path}")
    return faiss.read_index(path)


def load_model(hf_name: str) -> SentenceTransformer:
    return SentenceTransformer(hf_name)


import torch
import torch.nn as nn


class LinearAdapter(nn.Module):
    # Neural adapter for aligning query and passage embeddings (LINAW)
    # Architecture: Linear -> LayerNorm -> ReLU (preserves dimensionality)
    # Trained with triplet loss to improve retrieval on Tagalog/code-switched queries
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=True)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.norm(self.proj(x)))


def load_adapter(adapter_path: str, dim: int):
    # Load trained adapter weights; returns None if adapter file doesn't exist
    # Uses CPU loading to avoid GPU memory conflicts during multi-model serving
    if not os.path.exists(adapter_path):
        return None

    checkpoint = torch.load(adapter_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    adapter = LinearAdapter(dim)
    adapter.load_state_dict(state_dict, strict=True)
    adapter.eval()  # Disable dropout for inference
    return adapter


def encode_query(encoder, query_text, adapter=None):
    query_vector = encoder.encode(
        [query_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)

    if adapter is None:
        return query_vector

    with torch.no_grad():
        q = torch.from_numpy(query_vector)
        q = adapter(q)
        # L2 normalization required after adapter for IndexFlatIP (inner product) compatibility
        # IndexFlatIP expects normalized vectors to compute cosine similarity as dot product
        q = nn.functional.normalize(q, p=2, dim=-1)
        return q.cpu().numpy().astype(np.float32)


def faiss_idx_to_passage_id(faiss_idx: int, id_mapping: list[str]) -> str:
    # Map FAISS internal index to passage ID using id_mapping
    # Returns raw index as fallback if mapping is corrupted or out of bounds
    if 0 <= faiss_idx < len(id_mapping):
        return str(id_mapping[faiss_idx])
    return str(faiss_idx)


def run_search_sync(
    query: str,
    model_key: str,
    passages_df: pd.DataFrame,
    top_k: int,
    is_aligned: bool = False,
    language: str | None = None,
    document_type: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    if model_key not in MODEL_CONFIG:
        raise ValueError(f"Unknown model: {model_key}")

    cfg = MODEL_CONFIG[model_key]
    if model_key not in state.models:
        state.models[model_key] = load_model(cfg["hf_name"])

    model = state.models[model_key]
    index = state.faiss_indices[model_key]
    id_mapping = state.id_mapping

    # Load adapter if needed
    adapter = None
    if is_aligned:
        if model_key not in state.adapters:
            adapter_path = cfg.get("adapter_file")
            if adapter_path:
                state.adapters[model_key] = load_adapter(adapter_path, index.d)
        adapter = state.adapters.get(model_key)

    # Detect query language
    detected_language = identify_language(query.strip())

    # Encode query with optional adapter
    vec = encode_query(model, query.strip(), adapter=adapter)
    # Retrieve 3x top_k to allow post-filtering by language/type/date
    # Filtered results may reduce count below requested top_k
    distances, indices_result = index.search(vec, top_k * 3)

    results: list[dict[str, Any]] = []
    for faiss_idx, score in zip(indices_result[0], distances[0]):
        passage_id = faiss_idx_to_passage_id(int(faiss_idx), id_mapping)
        row = passages_df[passages_df["passage_id"] == passage_id]

        if row.empty:
            # Edge case: FAISS returned an index not present in passages CSV
            # Can occur if corpus was re-encoded without rebuilding FAISS index
            text = "_Passage not found in corpus._"
            short_title = None
            document_title = None
            display_title = "Unknown Document"
            lang = "—"
            doc_type = "—"
            date_filed = None
            source_url = None
        else:
            r = row.iloc[0]
            text = r.get("passage_text", r.get("text", "N/A"))
            short_title, document_title = passage_titles(r)
            lang = r.get("language", r.get("lang", "—"))
            doc_type = r.get("document_type", r.get("doc_type", "—"))
            date_filed = r.get("date_filed", None)
            source_url = cell_str(r.get("source_url"))
            display_title = resolve_display_title(
                short_title=short_title,
                document_title=document_title,
                passage_text=str(text),
                document_type=str(doc_type) if doc_type else None,
            )

        if language and lang != language and lang != "—":
            continue
        if document_type and doc_type != document_type and doc_type != "—":
            continue

        # Convert date_filed to string and normalize NaN/NaT values
        # Pandas may store dates as datetime objects or strings; handle both
        date_filed_str: str | None = None
        if date_filed and str(date_filed).lower() not in ("nan", "nat", "none", ""):
            date_filed_str = str(date_filed)

        if date_filed_str and date_from and date_filed_str < date_from:
            continue
        if date_filed_str and date_to and date_filed_str > date_to:
            continue

        # Confidence thresholds based on cosine similarity (0-1 range)
        # 0.65+ = high confidence (strong semantic match)
        # 0.50-0.65 = medium confidence (moderate match)
        # <0.50 = low confidence (weak or irrelevant)
        confidence = "HIGH" if score >= 0.65 else "MEDIUM" if score >= 0.50 else "LOW"

        results.append(
            {
                "rank": len(results) + 1,
                "passage_id": passage_id,
                "score": float(score),
                "confidence": confidence,
                "display_title": display_title,
                "document_title": document_title,
                "passage_text": str(text),
                "language": str(lang),
                "document_type": str(doc_type),
                "date_filed": date_filed_str,
                "source_url": source_url,
                "model": model_key,
                "query": query,
                "short_title": short_title,
            }
        )

        if len(results) >= top_k:
            break

    return results, detected_language


def load_citation_metadata() -> dict[str, dict[str, str | None]]:
    return load_doc_metadata_by_url(MASTER_CORPUS_CSV)
