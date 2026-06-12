"""Application paths and model configuration."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"
FRONTEND_ASSETS = FRONTEND_DIR / "assets"
PASSAGES_CSV = ROOT / "data/processed/module1_processed_passages.csv"
MASTER_CORPUS_CSV = ROOT / "data/raw/master_legal_corpus.csv"
METRICS_CSV = ROOT / "data/processed/detailed_evaluation_metrics.csv"
METRICS_BASELINE_CSV = ROOT / "data/processed/summary_evaluation_metrics_baseline.csv"
METRICS_ALIGNED_CSV = ROOT / "data/processed/summary_evaluation_metrics_aligned.csv"
METRICS_DETAILED_BASELINE_CSV = ROOT / "data/processed/detailed_evaluation_metrics_baseline.csv"
METRICS_DETAILED_ALIGNED_CSV = ROOT / "data/processed/detailed_evaluation_metrics_aligned.csv"
INDICES_DIR = ROOT / "data/processed/indices"

MODEL_CONFIG: dict[str, dict[str, str]] = {
    "BGE-M3": {
        "label": "BGE-M3 — BAAI General Embedding (Recommended)",
        "hf_name": "BAAI/bge-m3",
        "index_file": str(INDICES_DIR / "bge_m3_index.faiss"),
        "adapter_file": str(INDICES_DIR / "bge_m3_adapter.pt"),
    },
    "mSBERT": {
        "label": "mSBERT — Multilingual SBERT",
        "hf_name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "index_file": str(INDICES_DIR / "msbert_index.faiss"),
        "adapter_file": str(INDICES_DIR / "msbert_adapter.pt"),
    },
    "Legal-BERT": {
        "label": "Legal-BERT — Domain-Specific BERT",
        "hf_name": "nlpaueb/legal-bert-base-uncased",
        "index_file": str(INDICES_DIR / "legal_bert_index.faiss"),
        "adapter_file": str(INDICES_DIR / "legal_bert_adapter.pt"),
    },
}

DEFAULT_TOP_K = 5
DEFAULT_MODEL = "BGE-M3"
DEV_MODE = __import__("os").environ.get("DEV", "0") == "1"
