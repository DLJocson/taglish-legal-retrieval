"""Inspect citation_information keys in the raw master corpus (dev utility)."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data/raw/master_legal_corpus.csv"


def main() -> None:
    df = pd.read_csv(CSV)
    for i in range(min(3, len(df))):
        try:
            citation = json.loads(df.iloc[i]["citation_information"])
            print(f"Document {i} ({df.iloc[i]['label']}):")
            for key, value in citation.items():
                preview = str(value)[:100] if value else value
                print(f"  {key}: {preview}")
            print()
        except Exception as e:
            print(f"Document {i}: Error — {e}\n")


if __name__ == "__main__":
    main()
