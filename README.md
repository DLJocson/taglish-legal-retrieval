# Language vs Law: A Neural Alignment Adapter for Taglish Legal Query Retrieval in Philippine Statutes

*A retrieval system addressing Taglish query-document embedding mismatch via a lightweight neural alignment adapter.*

---

## 📋 Project Metadata

| Field | Details |
|-------|---------|
| **Institution** | Polytechnic University of the Philippines |
| **College** | College of Computer and Information Sciences, Manila, Philippines |
| **Department** | Department of Computer Science |
| **Section** | 3-2 |
| **Course** | COSC 304 — Introduction to Artificial Intelligence |
| **Members** | Cabbadu · Jocson · Lambohon · Salgado |
| **Section** | BSCS 3-2 |
| **Academic Year** | 2025–2026 |
| **Group** | Group 5 |

---

## Overview

LINAW (Legal Information Navigation and Alignment for the Web) addresses the fundamental register mismatch between informal Taglish queries and formal English legal documents in Philippine statutes. The system employs a parameter-efficient neural alignment adapter trained atop frozen encoder embeddings, enabling effective retrieval despite the linguistic gap between user queries and legal text.

---

## Core Features

- **Philippine Legal Retrieval Benchmark (PLRB)** — Newly constructed benchmark with 2,400 source documents, 28,970 retrieval passages, and 180 researcher-constructed queries
- **Neural Alignment Adapter** — Lightweight architecture using linear projection, layer normalization, and ReLU activation
- **Model Training** — Adapter optimized using contrastive triplet margin loss
- **Multi-language Evaluation** — Precision@5, Precision@10, and MRR metrics across English, Tagalog, and Taglish queries
- **Interactive Search UI** — Natural-language queries with language, document type, and date filters
- **Reproducible Pipeline** — Scripts for download, preprocessing, encoding, query annotation, and evaluation (`src/`)

---

## Tools and Technologies

- Python 3.12
- Sentence-Transformers (batched vectorization)
- FAISS (exact inner-product indexing)
- Pandas/PyArrow (OmniCorpus .parquet ingestion)
- Langdetect with custom heuristic (language identification)
- OpenAI API / GPT-4o (relevance annotation)
- MLflow (experiment tracking for runtimes, memory, and metrics)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12
- Sufficient disk space for models and processed data (~2.2 GB for BGE-M3 on first run)

### Installation

```bash
git clone https://github.com/DLJocson/taglish-legal-retrieval
cd taglish-legal-retrieval
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

- On first startup, **BGE-M3** (~2.2 GB), the parameter-efficient neural alignment adapter, and FAISS indices load into memory; other models load on first use.
- Frontend changes: save files, then **refresh the browser** (`DEV=1` disables caching).
- Python/API changes: `python run_dev.py --reload-backend`

---

## Basic Usage

### Web interface

1. Open the search page and enter a legal query (e.g., a statute topic or case issue).
2. Select an embedding model and optional filters.
3. Review ranked passages with scores and citation metadata.
4. Evaluate the Taglish-to-English alignment capabilities by observing how well the neural adapter bridges informal queries with formal legal documents.
5. Open **Analytics** after running `src/06_evaluate_retrieval.py` to view benchmark metrics.

### API example

```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"Ano ang penalties para sa estafa under the Revised Penal Code?\", \"model\": \"BGE-M3\", \"top_k\": 5}"
```

### Offline pipeline (summary)

The complete data pipeline is executed via the consolidated Jupyter notebook (`notebooks/linaw_pipeline.ipynb`), which includes:

- **Preprocessing** — Unicode NFKC normalization and `langdetect`
- **Encoding** — Creating L2-normalized embeddings
- **FAISS Indexing** — Using `IndexFlatIP` index
- **Evaluation** — Retrieval evaluation with PLRB

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
| `data/` | Philippine Legal Retrieval Benchmark (PLRB) dataset |
| `notebooks/` | Consolidated Jupyter notebook (publicly released artifact) |

See `frontend/README.md` for UI structure and development notes.

---

*Final project in partial fulfillment of COSC 304 — Introduction to Artificial Intelligence, PUP CCIS, AY 2025–2026. Language vs Law: A neural alignment adapter for Taglish legal query retrieval in Philippine statutes.*
