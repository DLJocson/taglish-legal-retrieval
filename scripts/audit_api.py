#!/usr/bin/env python3
"""Smoke-test all API endpoints. Run while server is up: python scripts/audit_api.py"""

from __future__ import annotations

import sys

import pandas as pd
import requests

BASE = "http://127.0.0.1:8000"
QUERY = "rights of an accused"


def main() -> int:
    failed: list[str] = []

    def check(name: str, fn) -> None:
        # Run a test function and track failures
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:
            print(f"FAIL  {name}: {e}")
            failed.append(name)

    # Smoke tests for all API endpoints
    check("GET /api/health", lambda: _health())
    check("GET /api/models", lambda: _models())
    check("GET /api/filters", lambda: _filters())
    for model in ("BGE-M3", "mSBERT", "Legal-BERT"):
        check(f"POST /api/search ({model})", lambda m=model: _search(m))
    check("POST /api/compare", _compare)
    check("POST /api/compare same-model returns 400", _compare_reject)
    check("POST /api/search + language filter", _search_filtered)
    check("GET /api/analytics/metrics", _analytics)
    check("analytics vs CSV", _analytics_accuracy)

    print()
    if failed:
        print(f"{len(failed)} check(s) failed.")
        return 1
    print("All checks passed.")
    return 0


def _health() -> None:
    # Verify server is running and corpus is loaded
    r = requests.get(f"{BASE}/api/health", timeout=15)
    r.raise_for_status()
    assert r.json()["passage_count"] > 0


def _models() -> None:
    # Verify all expected models are configured
    r = requests.get(f"{BASE}/api/models", timeout=15)
    keys = {m["key"] for m in r.json()["models"]}
    assert keys == {"BGE-M3", "mSBERT", "Legal-BERT"}


def _filters() -> None:
    # Verify filter options are populated from corpus
    r = requests.get(f"{BASE}/api/filters", timeout=15)
    d = r.json()
    assert "All" in d["languages"] and len(d["document_types"]) > 1


def _search(model: str) -> None:
    # Verify search returns correct number of results with metadata
    r = requests.post(
        f"{BASE}/api/search",
        json={"query": QUERY, "model": model, "top_k": 3},
        timeout=300,
    )
    r.raise_for_status()
    d = r.json()
    assert d["count"] == 3 and d["results"][0].get("display_title")


def _compare() -> None:
    # Verify comparison endpoint returns results for both models
    r = requests.post(
        f"{BASE}/api/compare",
        json={"query": "bail", "model_a": "BGE-M3", "model_b": "mSBERT", "top_k": 2},
        timeout=300,
    )
    r.raise_for_status()
    d = r.json()
    assert len(d["results_a"]) == len(d["results_b"]) == 2


def _compare_reject() -> None:
    # Verify comparison rejects identical model+adapter state (meaningless comparison)
    r = requests.post(
        f"{BASE}/api/compare",
        json={"query": "bail", "model_a": "BGE-M3", "model_b": "BGE-M3", "top_k": 2},
        timeout=30,
    )
    assert r.status_code == 400


def _search_filtered() -> None:
    # Verify language filter correctly excludes non-English results
    r = requests.post(
        f"{BASE}/api/search",
        json={"query": "school", "model": "BGE-M3", "top_k": 5, "language": "English"},
        timeout=120,
    )
    r.raise_for_status()
    for row in r.json()["results"]:
        assert row["language"] in ("English", "—")


def _analytics() -> None:
    # Verify analytics endpoint returns structured metrics data
    r = requests.get(f"{BASE}/api/analytics/metrics", timeout=30)
    r.raise_for_status()
    d = r.json()
    assert len(d["global"]) >= 3 and d["columns"]["mrr"]


def _analytics_accuracy() -> None:
    # Verify API metrics match manually calculated CSV values
    # Ensures backend aggregation logic is correct
    from pathlib import Path

    csv = Path(__file__).resolve().parent.parent / "data/processed/detailed_evaluation_metrics.csv"
    df = pd.read_csv(csv)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    api = requests.get(f"{BASE}/api/analytics/metrics", timeout=30).json()
    cols = api["columns"]
    metric_map = {"mrr": "mrr", "p5": "p@5", "p10": "p@10", "recall": "recall@10"}
    for api_key, csv_key in metric_map.items():
        manual = df.groupby("model")[csv_key].mean()
        api_vals = {r[cols["model"]]: r[cols[api_key]] for r in api["global"]}
        for model, val in api_vals.items():
            assert abs(val - manual[model]) < 1e-6, f"{api_key} {model}"


if __name__ == "__main__":
    sys.exit(main())
