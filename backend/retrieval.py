"""FAISS retrieval, corpus loading, and passage metadata."""

from __future__ import annotations

import json
import os
from typing import Any

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from backend.config import INDICES_DIR, MASTER_CORPUS_CSV, MODEL_CONFIG, PASSAGES_CSV
from backend.display_title import load_doc_metadata_by_url, resolve_display_title


class AppState:
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


def cell_str(value: Any) -> str | None:
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


def faiss_idx_to_passage_id(faiss_idx: int, id_mapping: list[str]) -> str:
    if 0 <= faiss_idx < len(id_mapping):
        return str(id_mapping[faiss_idx])
    return str(faiss_idx)


def run_search_sync(
    query: str,
    model_key: str,
    passages_df: pd.DataFrame,
    top_k: int,
    language: str | None = None,
    document_type: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict[str, Any]]:
    if model_key not in MODEL_CONFIG:
        raise ValueError(f"Unknown model: {model_key}")

    cfg = MODEL_CONFIG[model_key]
    if model_key not in state.models:
        state.models[model_key] = load_model(cfg["hf_name"])

    model = state.models[model_key]
    index = state.faiss_indices[model_key]
    id_mapping = state.id_mapping

    vec = model.encode([query.strip()], normalize_embeddings=True)
    vec = np.array(vec).astype("float32")
    distances, indices_result = index.search(vec, top_k * 3)

    results: list[dict[str, Any]] = []
    for faiss_idx, score in zip(indices_result[0], distances[0]):
        passage_id = faiss_idx_to_passage_id(int(faiss_idx), id_mapping)
        row = passages_df[passages_df["passage_id"] == passage_id]

        if row.empty:
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
        date_filed_str: str | None = None
        if date_filed and str(date_filed).lower() not in ("nan", "nat", "none", ""):
            date_filed_str = str(date_filed)

        if date_filed_str and date_from and date_filed_str < date_from:
            continue
        if date_filed_str and date_to and date_filed_str > date_to:
            continue

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

    return results


def load_citation_metadata() -> dict[str, dict[str, str | None]]:
    return load_doc_metadata_by_url(MASTER_CORPUS_CSV)
