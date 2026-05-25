# comparison-of-embedding-models-ph-legal-text
Comparative Evaluation of Embedding Models for Philippine Legal Text Retrieval. A final project in partial fulfillment for the course: COSC 304 – Introduction to Artificial Intelligence. PUP CCIS AY 2025-2026, BSCS 3-2, Group 5. Group Members: Cabbadu, Jocson, Lambohon, Salgado.

## Web app

```bash
pip install -r requirements.txt
python run_dev.py
```

- Search: http://127.0.0.1:8000/
- Analytics: http://127.0.0.1:8000/analytics

On first startup, BGE-M3 (~2.2GB) and FAISS indices load into memory once. Other embedding models load on first use.

Frontend edits apply after a browser refresh (`run_dev.py` sets `DEV=1` no-cache). Use `python run_dev.py --reload-backend` when changing Python code.

## Project layout

| Path | Purpose |
|------|---------|
| `api.py` | Uvicorn entry (`uvicorn api:app`) |
| `backend/` | FastAPI app, retrieval, display titles |
| `frontend/` | HTML + `assets/` (CSS/JS) |
| `src/` | Data pipeline scripts |
| `scripts/` | Dev utilities |
| `run_dev.py` | Local dev server |

See `frontend/README.md` for UI details.
