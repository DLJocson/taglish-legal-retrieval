"""Normalize, segment, and language-tag the PLRB master legal corpus.

Reads ``data/raw/master_legal_corpus.csv``, applies Unicode normalization and
sliding-window chunking, detects language per passage, assigns unique passage
IDs, and writes ``data/processed/module1_processed_passages.csv``.
"""

import json
import os
import re
import unicodedata

import pandas as pd
from langdetect import detect_langs


def normalize_text(text):
    """Normalize Unicode and collapse whitespace for consistent downstream use.

    Args:
        text: Raw document or passage text.

    Returns:
        NFKC-normalized string with single spaces, or empty string if input is
        not a string.
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFKC', text)
    return re.sub(r'\s+', ' ', text).strip()


def segment_passages(text, window=300, stride=50):
    """Split text into overlapping token windows for dense retrieval.

    Args:
        text: Full document text after normalization.
        window: Maximum tokens per passage (default 300).
        stride: Tokens advanced between windows (default 50).

    Returns:
        List of passage strings with overlap between consecutive chunks.
    """
    tokens = text.split()
    return [" ".join(tokens[i:i + window]) for i in range(0, len(tokens), window - stride)]


def identify_language(text):
    """Classify passage language, including English–Filipino code-switching.

    Code-switching is assigned when both English and Tagalog probabilities
    exceed 0.20; otherwise the dominant language is used.

    Args:
        text: Passage text to classify.

    Returns:
        One of ``English``, ``Filipino``, ``Code-Switched``, ``Other``, or
        ``Unknown`` if detection fails.
    """
    try:
        predictions = detect_langs(text)
        res = {l.lang: l.prob for l in predictions}

        if 'en' in res and 'tl' in res and min(res['en'], res['tl']) > 0.20:
            return "Code-Switched"

        dominant = predictions[0].lang
        return "Filipino" if dominant == 'tl' else "English" if dominant == 'en' else "Other"
    except:
        return "Unknown"


print("Loading perfectly cleaned master corpus for preprocessing...")
df = pd.read_csv("data/raw/master_legal_corpus.csv")
final_data = []

print(f"Normalizing, chunking, and language-tagging {len(df)} documents. This will take a few minutes...")

for index, doc in df.iterrows():
    # Extract metadata from citation_information JSON
    date_str = None
    title_str = None
    short_title_str = None
    try:
        citation = json.loads(doc.get('citation_information', '{}'))
        date_str = citation.get('date_of_enactment', None)
        title_str = citation.get('title', None)
        short_title_str = citation.get('short_title', None)
    except:
        pass
    
    clean_text = normalize_text(doc['text'])
    passages = segment_passages(clean_text)

    for chunk in passages:
        final_data.append({
            "source_url": doc.get('url', 'Unknown'),
            "document_type": doc.get('label', 'Unknown'),
            "document_title": title_str,
            "short_title": short_title_str,
            "date_filed": date_str,
            "passage_text": chunk,
            "language": identify_language(chunk)
        })

os.makedirs("data/processed", exist_ok=True)

processed_df = pd.DataFrame(final_data)

print("Generating unique Passage IDs...")
processed_df.insert(0, 'passage_id', ['PASSAGE_' + str(i).zfill(5) for i in range(len(processed_df))])

processed_df.to_csv("data/processed/module1_processed_passages.csv", index=False)

print("\n========================================")
print("   PROGRESS CHECK 1: FINAL DATA SUMMARY")
print("========================================")
print(f"Total Original Documents : {len(df)}")
print(f"Total Passages Generated : {len(processed_df)}")
print("\n--- Language Distribution ---")
print(processed_df['language'].value_counts())
print("========================================")
print("\nStep 2 Complete: Processed passages saved to data/processed/module1_processed_passages.csv")
