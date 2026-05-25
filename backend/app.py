"""FastAPI application: API routes, frontend static files, dev no-cache."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.config import (
    DEFAULT_MODEL,
    DEV_MODE,
    FRONTEND_ASSETS,
    FRONTEND_DIR,
    METRICS_CSV,
    MODEL_CONFIG,
)
from backend.retrieval import (
    load_citation_metadata,
    load_faiss_index,
    load_id_mapping,
    load_model,
    load_passages,
    run_search_sync,
    state,
)
from backend.schemas import CompareQuery, CompareResponse, SearchQuery, SearchResponse

_FRONTEND_NO_CACHE = (".html", ".css", ".js", ".map")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading passage corpus...")
    state.passages_df = load_passages()

    print("Loading document citation metadata (by source URL)...")
    state.doc_metadata_by_url = load_citation_metadata()
    print(f"  [ok] {len(state.doc_metadata_by_url)} source documents indexed")

    cols = state.passages_df.columns.tolist() if not state.passages_df.empty else []
    state.lang_col = next((c for c in cols if "lang" in c.lower()), None)
    state.type_col = next(
        (c for c in cols if "type" in c.lower() or "doc_type" in c.lower()),
        None,
    )

    print("Loading ID mapping...")
    state.id_mapping = load_id_mapping()

    print("Loading FAISS indices...")
    for key, cfg in MODEL_CONFIG.items():
        state.faiss_indices[key] = load_faiss_index(cfg["index_file"])
        print(f"  [ok] {key} index ready")

    print(f"Loading default model ({DEFAULT_MODEL}); this may take a minute...")
    default_cfg = MODEL_CONFIG[DEFAULT_MODEL]
    state.models[DEFAULT_MODEL] = await asyncio.to_thread(
        load_model, default_cfg["hf_name"]
    )
    print(f"  [ok] {DEFAULT_MODEL} loaded")

    yield

    state.models.clear()
    state.faiss_indices.clear()


def create_app() -> FastAPI:
    application = FastAPI(
        title="PH Legal AI Search API",
        description="Dense retrieval over Philippine legal corpora",
        version="1.0.0",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if DEV_MODE:

        @application.middleware("http")
        async def disable_frontend_cache(request: Request, call_next):
            response = await call_next(request)
            path = request.url.path
            if path in ("/", "/analytics") or path.startswith("/assets") or any(
                path.endswith(ext) for ext in _FRONTEND_NO_CACHE
            ):
                response.headers["Cache-Control"] = "no-store, must-revalidate"
                response.headers["Pragma"] = "no-cache"
            return response

    @application.get("/ping")
    async def ping():
        return {"status": "pong"}

    @application.get("/api/health")
    async def health():
        return {
            "status": "ok",
            "models_loaded": list(state.models.keys()),
            "indices_loaded": list(state.faiss_indices.keys()),
            "passage_count": len(state.passages_df),
            "dev_mode": DEV_MODE,
        }

    @application.get("/api/models")
    async def list_models():
        return {
            "models": [
                {"key": k, "label": v["label"], "loaded": k in state.models}
                for k, v in MODEL_CONFIG.items()
            ],
            "default": DEFAULT_MODEL,
        }

    @application.get("/api/filters")
    async def get_filters():
        langs = ["All"]
        types = ["All"]
        date_from = None
        date_to = None

        if not state.passages_df.empty:
            if state.lang_col:
                langs += sorted(state.passages_df[state.lang_col].dropna().unique().tolist())
            if state.type_col:
                types += sorted(state.passages_df[state.type_col].dropna().unique().tolist())
            if "date_filed" in state.passages_df.columns:
                valid_dates = state.passages_df["date_filed"].dropna()
                if not valid_dates.empty:
                    date_from = str(valid_dates.min())
                    date_to = str(valid_dates.max())

        return {
            "languages": langs,
            "document_types": types,
            "date_from": date_from,
            "date_to": date_to,
        }

    @application.post("/api/search", response_model=SearchResponse)
    async def search(body: SearchQuery):
        results = await asyncio.to_thread(
            run_search_sync,
            body.query,
            body.model,
            state.passages_df,
            body.top_k,
            body.language if body.language != "All" else None,
            body.document_type if body.document_type != "All" else None,
            body.date_from,
            body.date_to,
        )
        return SearchResponse(
            results=results,
            query=body.query,
            model=body.model,
            count=len(results),
        )

    @application.post("/api/compare", response_model=CompareResponse)
    async def compare(body: CompareQuery):
        if body.model_a == body.model_b:
            raise HTTPException(
                status_code=400,
                detail="Select two different models for a meaningful comparison.",
            )

        results_a, results_b = await asyncio.gather(
            asyncio.to_thread(
                run_search_sync,
                body.query,
                body.model_a,
                state.passages_df,
                body.top_k,
            ),
            asyncio.to_thread(
                run_search_sync,
                body.query,
                body.model_b,
                state.passages_df,
                body.top_k,
            ),
        )

        return CompareResponse(
            query=body.query,
            model_a=body.model_a,
            model_b=body.model_b,
            results_a=results_a,
            results_b=results_b,
        )

    @application.get("/api/analytics/metrics")
    async def analytics_metrics():
        if not METRICS_CSV.exists():
            raise HTTPException(
                status_code=404,
                detail=(
                    "Evaluation data not found. Run 06_evaluate_retrieval.py to generate "
                    "data/processed/detailed_evaluation_metrics.csv"
                ),
            )

        df = pd.read_csv(METRICS_CSV)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        all_cols = df.columns.tolist()
        model_col = next((c for c in all_cols if "model" in c), None)
        lang_col = next((c for c in all_cols if "lang" in c), None)
        sem_col = next((c for c in all_cols if "semantic" in c or "type" in c), None)
        mrr_col = next((c for c in all_cols if "mrr" in c), None)
        p5_col = next((c for c in all_cols if "p@5" in c or "p5" in c), None)
        p10_col = next(
            (c for c in all_cols if any(p in c for p in ["p@10", "p_10", "p10"])),
            None,
        )
        recall_col = next((c for c in all_cols if "recall" in c), None)

        if not model_col:
            raise HTTPException(status_code=500, detail="CSV missing a model column")

        agg_cols = {c: "mean" for c in [mrr_col, p5_col, p10_col, recall_col] if c}
        global_df = df.groupby(model_col).agg(agg_cols).reset_index()

        def _serialize_frame(frame: pd.DataFrame) -> list[dict]:
            out = frame.copy()
            for col in out.select_dtypes(include=["number"]).columns:
                out[col] = out[col].astype(float)
            return out.to_dict(orient="records")

        payload: dict[str, Any] = {
            "columns": {
                "model": model_col,
                "language": lang_col,
                "semantic": sem_col,
                "mrr": mrr_col,
                "p5": p5_col,
                "p10": p10_col,
                "recall": recall_col,
            },
            "global": _serialize_frame(global_df),
            "raw_count": len(df),
        }

        if lang_col and agg_cols:
            lang_agg = df.groupby([model_col, lang_col]).agg(agg_cols).reset_index()
            payload["by_language"] = _serialize_frame(lang_agg)

        if sem_col and agg_cols:
            sem_agg = df.groupby([model_col, sem_col]).agg(agg_cols).reset_index()
            payload["by_semantic"] = _serialize_frame(sem_agg)

        payload["raw_sample"] = _serialize_frame(df.head(500))
        return payload

    if FRONTEND_ASSETS.is_dir():
        application.mount(
            "/assets",
            StaticFiles(directory=str(FRONTEND_ASSETS)),
            name="assets",
        )

    @application.get("/", response_class=FileResponse)
    async def serve_index():
        path = FRONTEND_DIR / "index.html"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="frontend/index.html not found")
        return FileResponse(path, headers=_no_cache_headers())

    @application.get("/analytics", response_class=FileResponse)
    async def serve_analytics():
        path = FRONTEND_DIR / "analytics.html"
        if not path.is_file():
            raise HTTPException(
                status_code=404, detail="frontend/analytics.html not found"
            )
        return FileResponse(path, headers=_no_cache_headers())

    return application


def _no_cache_headers() -> dict[str, str]:
    if not DEV_MODE:
        return {}
    return {"Cache-Control": "no-store, must-revalidate", "Pragma": "no-cache"}


app = create_app()
