# Comparative Evaluation of Embedding Models for Philippine Legal Text Retrieval

*A retrieval benchmarking study and interactive search demo for Philippine legal corpora.*

---

## 📋 Project Metadata

| Field | Details |
|-------|---------|
| **Institution** | Polytechnic University of the Philippines |
| **College** | College of Computer and Information Sciences |
| **Department** | Department of Computer Science |
| **Course** | COSC 304 — Introduction to Artificial Intelligence |
| **Section** | BSCS 3-2 |
| **Academic Year** | 2025–2026 |
| **Group** | Group 5 |

**Members:** Cabbadu · Jocson · Lambohon · Salgado

---

## Overview

This project compares **embedding models** for semantic search over Philippine legal text. It includes an offline data pipeline (corpus preparation, indexing, stratified evaluation) and a **FastAPI web application** for live search, side-by-side model comparison, and analytics dashboards.

---

## Core Features

- **Multi-model retrieval** — BGE-M3, multilingual SBERT (mSBERT), and Legal-BERT over FAISS indices
- **Interactive search UI** — natural-language queries with language, document type, and date filters
- **Model comparison** — run the same query on two models and inspect ranked passages side by side
- **Evaluation analytics** — MRR, nDCG, and related metrics from stratified retrieval benchmarks
- **Reproducible pipeline** — scripts for download, preprocessing, encoding, query annotation, and evaluation (`src/`)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+ (recommended)
- Sufficient disk space for models and processed data (~2.2 GB for BGE-M3 on first run)

### Installation

```bash
git clone https://github.com/<your-org>/comparison-of-embedding-models-ph-legal-text.git
cd comparison-of-embedding-models-ph-legal-text
pip install -r requirements.txt
```

### Run the web app

```bash
python run_dev.py
```

| Page | URL |
|------|-----|
| Search | http://127.0.0.1:8000/ |
| Analytics | http://127.0.0.1:8000/analytics |

**Notes**

- On first startup, **BGE-M3** (~2.2 GB) and FAISS indices load into memory; other models load on first use.
- Frontend changes: save files, then **refresh the browser** (`DEV=1` disables caching).
- Python/API changes: `python run_dev.py --reload-backend`

---

## Basic Usage

### Web interface

1. Open the search page and enter a legal query (e.g., a statute topic or case issue).
2. Select an embedding model and optional filters.
3. Review ranked passages with scores and citation metadata.
4. Use **Compare** to contrast two models on the same query.
5. Open **Analytics** after running `src/06_evaluate_retrieval.py` to view benchmark metrics.

### API example

```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"prescription of actions\", \"model\": \"BGE-M3\", \"top_k\": 5}"
```

### Offline pipeline (summary)

| Stage | Script |
|-------|--------|
| Download corpus | `src/01_download_omnicorpus.py` |
| Preprocess & segment | `src/02_preprocess_and_segment.py` |
| Build tables | `src/03_generate_table.py` |
| Encode & index | `src/04a_encode_corpus.py` |
| Build queries | `src/04b_build_queries_csv.py` |
| Evaluate retrieval | `src/06_evaluate_retrieval.py` |

---

## Project Layout

| Path | Purpose |
|------|---------|
| `api.py` | Uvicorn entry (`uvicorn api:app`) |
| `backend/` | FastAPI app, retrieval, schemas |
| `frontend/` | HTML, CSS, JS (search & analytics) |
| `src/` | Data pipeline and evaluation |
| `scripts/` | Dev utilities |
| `run_dev.py` | Local development server |

See `frontend/README.md` for UI structure and development notes.

---

*Final project in partial fulfillment of COSC 304 — Introduction to Artificial Intelligence, PUP CCIS, AY 2025–2026.*
