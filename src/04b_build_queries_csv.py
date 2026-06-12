"""Build structured evaluation queries from raw text input.

Parses ``data/raw/raw_queries.txt``, assigns query IDs, detects language and
semantic type via heuristics, and writes ``data/processed/queries.csv`` with
empty relevance labels for the LLM annotator.
"""

import os

import pandas as pd
from langdetect import detect_langs

print("Automating queries.csv generation...")


def identify_language(text):
    """Detect query language using the same code-switching rule as preprocessing.

    Args:
        text: Raw query string.

    Returns:
        ``Code-Switched``, ``Tagalog``, ``English``, or ``Other``; defaults to
        ``English`` when detection fails.
    """
    try:
        predictions = detect_langs(text)
        res = {l.lang: l.prob for l in predictions}

        if 'en' in res and 'tl' in res and min(res['en'], res['tl']) > 0.20:
            return "Code-Switched"

        dominant = predictions[0].lang
        return "Tagalog" if dominant == 'tl' else "English" if dominant == 'en' else "Other"
    except:
        return "English"


def classify_semantic_type(text):
    """Assign a semantic category from keyword heuristics.

    Args:
        text: Raw query string.

    Returns:
        One of ``Definitional``, ``Procedural``, or ``Case-Law``.
    """
    text_lower = text.lower()

    procedural_keywords = ['how', 'paano', 'saan', 'process', 'remedy', 'gagawin', 'file', 'reklamo', 'steps', 'grace period', 'do i do', 'dapat gawin']

    definitional_keywords = ['what is', 'ano ang', 'meaning', 'elements', 'ibig sabihin', 'difference', 'distinction', 'law against', 'sakop ba', 'counted ba', 'penalty', 'parusa']

    if any(word in text_lower for word in definitional_keywords):
        return "Definitional"
    elif any(word in text_lower for word in procedural_keywords):
        return "Procedural"
    else:
        return "Case-Law"


with open("data/raw/raw_queries.txt", "r", encoding="utf-8") as f:
    raw_queries = [line.strip() for line in f.readlines() if line.strip()]

print(f"Loaded {len(raw_queries)} raw queries.")

formatted_data = []

for index, query in enumerate(raw_queries):
    query_id = f"Q_{str(index + 1).zfill(3)}"

    formatted_data.append({
        "query_id": query_id,
        "query_text": query,
        "language_label": identify_language(query),
        "semantic_type": classify_semantic_type(query),
        "relevant_passage_ids": ""
    })

df = pd.DataFrame(formatted_data)

# --- Stratified Train/Test Split ---
# This ensures 80% of queries go to training, 20% to testing, 
# balanced evenly across English, Tagalog, and Code-Switched.
test_df = df.groupby('language_label', group_keys=False).apply(lambda x: x.sample(frac=0.2, random_state=42))
df['split'] = 'train'
df.loc[test_df.index, 'split'] = 'test'
# ----------------------------------------

output_path = "data/processed/queries.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

print(f"\nSuccess! Structured CSV saved to {output_path}")
print("\nSplit Distribution:")
print(pd.crosstab(df['language_label'], df['split']))