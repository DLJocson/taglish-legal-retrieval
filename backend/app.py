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
    METRICS_DETAILED_ALIGNED_CSV,
    METRICS_BASELINE_CSV,
    METRICS_ALIGNED_CSV,
    MODEL_CONFIG,
)
from backend.retrieval import (
    get_demo_metrics,
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
    # Load all heavy resources at startup to avoid per-request latency
    print("Loading passage corpus...")
    state.passages_df = load_passages()

    print("Loading document citation metadata (by source URL)...")
    state.doc_metadata_by_url = load_citation_metadata()
    print(f"  [ok] {len(state.doc_metadata_by_url)} source documents indexed")

    # Detect column names for language and document type (may vary across CSV versions)
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

    # Load default model immediately; other models loaded lazily on first use
    print(f"Loading default model ({DEFAULT_MODEL}); this may take a minute...")
    default_cfg = MODEL_CONFIG[DEFAULT_MODEL]
    state.models[DEFAULT_MODEL] = await asyncio.to_thread(
        load_model, default_cfg["hf_name"]
    )
    print(f"  [ok] {DEFAULT_MODEL} loaded")

    yield

    # Clean up resources on shutdown
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
        # Disable browser caching in dev mode to ensure frontend changes are reflected immediately
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
        results, detected_language = await asyncio.to_thread(
            run_search_sync,
            body.query,
            body.model,
            state.passages_df,
            body.top_k,
            body.is_aligned,
            body.language if body.language != "All" else None,
            body.document_type if body.document_type != "All" else None,
            body.date_from,
            body.date_to,
        )
        demo_metrics = get_demo_metrics(body.query, body.model)
        return SearchResponse(
            results=results,
            query=body.query,
            model=body.model,
            count=len(results),
            detected_language=detected_language,
            demo_metrics=demo_metrics,
        )

    @application.post("/api/compare", response_model=CompareResponse)
    async def compare(body: CompareQuery):
        # Prevent meaningless comparisons: identical model + adapter state yields identical results
        if body.model_a == body.model_b and body.is_aligned_a == body.is_aligned_b:
            raise HTTPException(
                status_code=400,
                detail="Select different models or different adapter states for comparison.",
            )

        results_a, detected_lang_a = await asyncio.to_thread(
            run_search_sync,
            body.query,
            body.model_a,
            state.passages_df,
            body.top_k,
            body.is_aligned_a,
        )
        results_b, detected_lang_b = await asyncio.to_thread(
            run_search_sync,
            body.query,
            body.model_b,
            state.passages_df,
            body.top_k,
            body.is_aligned_b,
        )

        return CompareResponse(
            query=body.query,
            model_a=body.model_a,
            model_b=body.model_b,
            is_aligned_a=body.is_aligned_a,
            is_aligned_b=body.is_aligned_b,
            detected_language=detected_lang_a,  # Language detection is query-dependent, not model-dependent
            results_a=results_a,
            results_b=results_b,
        )

    @application.get("/api/analytics/metrics")
    async def analytics_metrics():
        # Aggregate evaluation metrics from CSV files for analytics dashboard
        # Computes global, language-specific, and semantic-type-specific performance
        if not METRICS_BASELINE_CSV.exists() or not METRICS_ALIGNED_CSV.exists():
            raise HTTPException(
                status_code=404,
                detail=(
                    "Evaluation data not found. Run 06_evaluate_retrieval.py to generate "
                    "summary_evaluation_metrics_baseline.csv and summary_evaluation_metrics_aligned.csv"
                ),
            )

        df_baseline = pd.read_csv(METRICS_BASELINE_CSV)
        df_aligned = pd.read_csv(METRICS_ALIGNED_CSV)

        # Normalize column names for consistent access (handle spacing/casing variations)
        df_baseline.columns = [c.strip().lower().replace(" ", "_") for c in df_baseline.columns]
        df_aligned.columns = [c.strip().lower().replace(" ", "_") for c in df_aligned.columns]

        all_cols = df_baseline.columns.tolist()
        model_col = next((c for c in all_cols if "model" in c), None)
        # Prefer "category" over "scope" for filtering (CSV format may vary)
        category_col = next((c for c in all_cols if "category" in c), None) or next((c for c in all_cols if "scope" in c), None)
        mrr_col = next((c for c in all_cols if "mrr" in c), None)
        p5_col = next((c for c in all_cols if "p@5" in c or "p5" in c), None)
        p10_col = next(
            (c for c in all_cols if any(p in c for p in ["p@10", "p_10", "p10"])),
            None,
        )
        recall_col = next((c for c in all_cols if "recall" in c), None)

        # Normalize category values to match frontend filtering logic (hyphens to underscores)
        if category_col:
            df_baseline[category_col] = df_baseline[category_col].str.lower().str.replace("-", "_")
            df_aligned[category_col] = df_aligned[category_col].str.lower().str.replace("-", "_")

        if not model_col:
            raise HTTPException(status_code=500, detail="CSV missing a model column")

        metric_cols = [c for c in [mrr_col, p5_col, p10_col, recall_col] if c]

        def _serialize_frame(frame: pd.DataFrame) -> list[dict]:
            # Convert DataFrame to JSON-serializable format with proper float handling
            out = frame.copy()
            for col in out.select_dtypes(include=["number"]).columns:
                out[col] = out[col].astype(float)
            return out.to_dict(orient="records")

        # Calculate global metrics (averaged across all queries) for baseline and aligned
        baseline_global = df_baseline[df_baseline[category_col] == "all"].groupby(model_col)[metric_cols].mean().reset_index()
        aligned_global = df_aligned[df_aligned[category_col] == "all"].groupby(model_col)[metric_cols].mean().reset_index()

        # Calculate deltas (aligned - baseline) to show LINAW improvement per model
        deltas = []
        for model in baseline_global[model_col].unique():
            base_row = baseline_global[baseline_global[model_col] == model]
            aligned_row = aligned_global[aligned_global[model_col] == model]

            if not base_row.empty and not aligned_row.empty:
                delta_row = {model_col: model}
                for col in metric_cols:
                    base_val = base_row[col].values[0] if not base_row.empty else 0
                    aligned_val = aligned_row[col].values[0] if not aligned_row.empty else 0
                    delta_row[f"{col}_delta"] = aligned_val - base_val
                deltas.append(delta_row)

        # Language-specific metrics: Code-Switched (primary target for LINAW)
        baseline_lang = df_baseline[df_baseline[category_col] == "code_switched"].groupby(model_col)[metric_cols].mean().reset_index()
        aligned_lang = df_aligned[df_aligned[category_col] == "code_switched"].groupby(model_col)[metric_cols].mean().reset_index()

        # Language-specific metrics: English (baseline for comparison)
        baseline_english = df_baseline[df_baseline[category_col] == "english"].groupby(model_col)[metric_cols].mean().reset_index()
        aligned_english = df_aligned[df_aligned[category_col] == "english"].groupby(model_col)[metric_cols].mean().reset_index()

        # Language-specific metrics: Tagalog (monolingual baseline)
        baseline_tagalog = df_baseline[df_baseline[category_col] == "tagalog"].groupby(model_col)[metric_cols].mean().reset_index()
        aligned_tagalog = df_aligned[df_aligned[category_col] == "tagalog"].groupby(model_col)[metric_cols].mean().reset_index()

        # Semantic type breakdown to analyze performance across query types
        # Case-Law: queries about judicial decisions
        # Definitional: queries asking for legal definitions
        # Procedural: queries about legal processes
        baseline_semantic = df_baseline[df_baseline[category_col].isin(["case_law", "definitional", "procedural"])].groupby([model_col, category_col])[metric_cols].mean().reset_index()
        aligned_semantic = df_aligned[df_aligned[category_col].isin(["case_law", "definitional", "procedural"])].groupby([model_col, category_col])[metric_cols].mean().reset_index()

        # Add per-query sample data from detailed metrics (limited to 500 rows for performance)
        sample_data = []
        if METRICS_DETAILED_ALIGNED_CSV.exists():
            df_detailed = pd.read_csv(METRICS_DETAILED_ALIGNED_CSV)
            df_detailed.columns = [c.strip().lower().replace(" ", "_") for c in df_detailed.columns]
            sample_data = _serialize_frame(df_detailed.head(500))

        payload: dict[str, Any] = {
            "columns": {
                "model": model_col,
                "category": category_col,
                "mrr": mrr_col,
                "p5": p5_col,
                "p10": p10_col,
                "recall": recall_col,
            },
            "baseline_global": _serialize_frame(baseline_global),
            "aligned_global": _serialize_frame(aligned_global),
            "deltas": deltas,
            "baseline_language": _serialize_frame(baseline_lang),
            "aligned_language": _serialize_frame(aligned_lang),
            "baseline_english": _serialize_frame(baseline_english),
            "aligned_english": _serialize_frame(aligned_english),
            "baseline_tagalog": _serialize_frame(baseline_tagalog),
            "aligned_tagalog": _serialize_frame(aligned_tagalog),
            "baseline_semantic": _serialize_frame(baseline_semantic),
            "aligned_semantic": _serialize_frame(aligned_semantic),
            "sample_data": sample_data,
        }

        return payload

    # Mount static assets directory for CSS/JS files
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
    # Return cache-control headers only in dev mode to ensure hot-reload works
    if not DEV_MODE:
        return {}
    return {"Cache-Control": "no-store, must-revalidate", "Pragma": "no-cache"}


app = create_app()
