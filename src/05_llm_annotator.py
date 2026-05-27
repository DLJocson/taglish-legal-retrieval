print("========================================")
print(" STAGE 2: LLM VERTEX AI ANNOTATOR  ")
print("========================================")

import os
import gc
import csv
import json
import shutil
import time
from itertools import zip_longest

import pandas as pd
import faiss
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer

# ==============================================================================
# EXACT BASE_PATH REGISTRATION
# ==============================================================================
BASE_PATH = r"C:\Users\louie\OneDrive\Documents\GitHub\comparison-of-embedding-models-ph-legal-text\data\processed"
print(f"[SETTING] Targeted absolute directory path: {BASE_PATH}")

# =========================
# CONFIGURATION
# =========================
MODEL_ID = "gemini-2.5-flash"
PROJECT_ID = "project-fc4bdbdc-0585-464a-864"
LOCATION = "us-central1"

QUERIES_CSV = os.path.join(BASE_PATH, "queries.csv")
QUERIES_BACKUP_CSV = os.path.join(BASE_PATH, "queries_original.csv")
PASSAGES_CSV = os.path.join(BASE_PATH, "module1_processed_passages.csv")
ID_MAPPING_JSON = os.path.join(BASE_PATH, "indices", "id_mapping.json")

START_FROM_INDEX = 0
TOP_K_PER_MODEL = 10
MAX_PASSAGES_IN_PROMPT = 30
MAX_CHARS_PER_PASSAGE = 1200
SAVE_EVERY = 1
MAX_RETRIES = 3

ALLOWED_LANGUAGES = {"English", "Tagalog", "Code-Switched"}

MODEL_CONFIGS = {
    "msbert": "paraphrase-multilingual-mpnet-base-v2",
    "bge_m3": "BAAI/bge-m3",
    "legal_bert": "nlpaueb/legal-bert-base-uncased",
}

# =========================
# UTILITIES & DATA PARSERS
# =========================
def clear_memory():
    gc.collect()

def parse_list_value(value):
    """Safely parse list-like CSV values."""
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text.replace("'", '"'))
                if isinstance(parsed, list):
                    return [str(v) for v in parsed if str(v).strip()]
            except Exception:
                try:
                    import ast
                    parsed = ast.literal_eval(text)
                    if isinstance(parsed, list):
                        return [str(v) for v in parsed if str(v).strip()]
                except Exception:
                    return []
    return []

def unique_preserve_order(items):
    seen = set()
    out = []
    for item in items:
        item = str(item).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out

def truncate_text(text, max_chars):
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."

def build_prompt(query_text, context):
    return f"""
You are a precise Philippine legal assistant.

QUERY:
"{query_text}"

RETRIEVED PASSAGES:
{context}

TASK:
Return STRICT JSON only. Do not wrap in markdown or backticks.

{{
  "passage_ids": ["PASSAGE_XXXXX"]
}}

RULES:
- Only include IDs from the retrieved passages shown above
- Include ALL relevant passages that answer the query
- If none are relevant, return: {{"passage_ids": []}}
- Do NOT explain anything
- Do NOT add extra keys
- Output MUST be raw JSON only
""".strip()

def safe_json_parse(text):
    try:
        text = (text or "").strip()
        if "```" in text:
            text = text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {"passage_ids": []}

def load_id_lookup(mapping_path):
    with open(mapping_path, "r") as f:
        mapping = json.load(f)

    if isinstance(mapping, list):
        def lookup(i):
            return str(mapping[i])
        return lookup

    if isinstance(mapping, dict) and len(mapping) > 0:
        sample_key = next(iter(mapping.keys()))
        sample_val = mapping[sample_key]

        if str(sample_key).isdigit():
            idx_map = {int(k): v for k, v in mapping.items()}
            def lookup(i):
                return str(idx_map[i])
            return lookup

        if isinstance(sample_val, int) or (isinstance(sample_val, str) and str(sample_val).isdigit()):
            reverse_map = {int(v): k for k, v in mapping.items()}
            def lookup(i):
                return str(reverse_map[i])
            return lookup

    raise ValueError("Unsupported id_mapping.json format.")

def serialize_sources(provenance, relevant_ids):
    return json.dumps(
        {pid: sorted(list(provenance.get(pid, []))) for pid in relevant_ids},
        ensure_ascii=False
    )

# =========================
# LOAD DATA
# =========================
print("Loading dataset...", flush=True)

if os.path.exists(QUERIES_BACKUP_CSV) and START_FROM_INDEX == 0:
    print("Restoring original dataset...", flush=True)
    shutil.copyfile(QUERIES_BACKUP_CSV, QUERIES_CSV)

# Defensive catch checks to prevent silent script crashes before execution loop
if not os.path.exists(QUERIES_CSV):
    raise FileNotFoundError(f"Missing absolute query target: {QUERIES_CSV}")
if not os.path.exists(PASSAGES_CSV):
    raise FileNotFoundError(f"Missing absolute corpus passages target: {PASSAGES_CSV}")

queries_df = pd.read_csv(QUERIES_CSV)

# --- ROBUST PASSAGE INGESTION LAYER (Resolves Tokenization EOF Mistakes) ---
print("Parsing corpus passages with quoting safety controls...", flush=True)
try:
    # Attempt 1: Standard read
    passages_df = pd.read_csv(PASSAGES_CSV)
except pd.errors.ParserError:
    try:
        # Attempt 2: Switch engine to Python and handle malformed quote strings gracefully
        print("  Standard parser encountered a syntax anomaly. Retrying with Python engine...", flush=True)
        passages_df = pd.read_csv(PASSAGES_CSV, engine='python', on_bad_lines='skip')
    except Exception:
        # Attempt 3: Disable strict quoting definitions completely to process strings literally
        print("  Line parsing error persisted. Activating literal fallback rules...", flush=True)
        passages_df = pd.read_csv(PASSAGES_CSV, quoting=csv.QUOTE_NONE, on_bad_lines='skip')
# --------------------------------------------------------------------------

required_cols = {"query_id", "query_text", "language_label", "semantic_type", "relevant_passage_ids"}
missing_cols = required_cols - set(queries_df.columns)
if missing_cols:
    raise ValueError(f"Missing required columns in queries CSV: {sorted(missing_cols)}")

queries_df["query_text"] = queries_df["query_text"].astype(str)
queries_df["language_label"] = queries_df["language_label"].astype(str).str.strip()
queries_df["semantic_type"] = queries_df["semantic_type"].astype(str).str.strip()
queries_df["relevant_passage_ids"] = queries_df["relevant_passage_ids"].apply(parse_list_value)

if "retrieval_sources" not in queries_df.columns:
    queries_df["retrieval_sources"] = ""
else:
    queries_df["retrieval_sources"] = queries_df["retrieval_sources"].astype(str)

queries_df["in_scope"] = queries_df["language_label"].isin(ALLOWED_LANGUAGES)
passages_df["passage_id"] = passages_df["passage_id"].astype(str)

id_lookup = load_id_lookup(ID_MAPPING_JSON)
passage_lookup = passages_df.set_index("passage_id")["passage_text"].to_dict()

in_scope_count = int(queries_df["in_scope"].sum())
out_of_scope_count = len(queries_df) - in_scope_count

print(f"Total queries: {len(queries_df)}")
print(f"In-scope queries: {in_scope_count}")
print(f"Out-of-scope queries excluded: {out_of_scope_count}")

# =========================
# LOAD RETRIEVERS
# =========================
print("Loading retrieval models (CPU)...", flush=True)

loaded_systems = {}
for alias, path in MODEL_CONFIGS.items():
    encoder = SentenceTransformer(path, device="cpu")
    index_path = f"{BASE_PATH}/indices/{alias}_index.faiss"
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Target multi-dimensional embedding map unreadable: {index_path}")
    index = faiss.read_index(index_path)
    loaded_systems[alias] = {"encoder": encoder, "index": index}

# =========================
# ANNOTATION LOOP
# =========================
for idx, row in queries_df.iloc[START_FROM_INDEX:].iterrows():
    if not bool(row["in_scope"]):
        print(f"Skipping out-of-scope query {row['query_id']} ({row['language_label']}).", flush=True)
        continue

    existing_ids = row["relevant_passage_ids"]
    if isinstance(existing_ids, list) and len(existing_ids) > 0:
        print(f"Skipping {row['query_id']} already annotated.", flush=True)
        continue

    query_text = str(row["query_text"])
    print(f"\n[{idx + 1}/{len(queries_df)}] Processing {row['query_id']}...", flush=True)

    results_per_model = []
    provenance = {}

    for alias, system in loaded_systems.items():
        print(f"  Retrieving with {alias.upper()}...", flush=True)

        query_vec = system["encoder"].encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        ).astype("float32")

        _, indices = system["index"].search(query_vec, TOP_K_PER_MODEL)

        model_passage_ids = []
        for fid in indices[0]:
            if fid == -1:
                continue
            try:
                pid = str(id_lookup(int(fid)))
                model_passage_ids.append(pid)
                provenance.setdefault(pid, set()).add(alias)
            except Exception as e:
                print(f"    Warning: could not map FAISS id {fid}: {e}", flush=True)

        results_per_model.append(model_passage_ids)

    # Interleave pooled results across models
    pooled_ids = []
    seen = set()
    for group in zip_longest(*results_per_model):
        for pid in group:
            if pid is None:
                continue
            pid = str(pid)
            if pid not in seen:
                seen.add(pid)
                pooled_ids.append(pid)

    pooled_ids = pooled_ids[:MAX_PASSAGES_IN_PROMPT]
    print(f"  Pooled passages: {pooled_ids}", flush=True)

    context = ""
    for pid in pooled_ids:
        text = passage_lookup.get(pid)
        if text:
            context += f"ID: {pid}\nTEXT: {truncate_text(text, MAX_CHARS_PER_PASSAGE)}\n\n"

    prompt = build_prompt(query_text, context)

    relevant_ids = []
    success = False

    # =========================
    # VERTEX GENAI WORKFLOW
    # =========================
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    for attempt in range(MAX_RETRIES):
        try:
            print(f"  Gemini attempt {attempt + 1}...", flush=True)
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                ),
            )

            parsed = safe_json_parse(response.text)
            raw_ids = parsed.get("passage_ids", [])
            raw_ids = [str(pid).strip() for pid in raw_ids]
            relevant_ids = unique_preserve_order([pid for pid in raw_ids if pid in pooled_ids])

            success = True
            print(f"  Success: {relevant_ids}", flush=True)
            break

        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {type(e).__name__}: {e}", flush=True)
            time.sleep((attempt + 1) * 5)

    if success:
        queries_df.at[idx, "relevant_passage_ids"] = json.dumps(relevant_ids, ensure_ascii=False)
        queries_df.at[idx, "retrieval_sources"] = serialize_sources(provenance, relevant_ids)
        print("  Saved row.", flush=True)

    if SAVE_EVERY == 1:
        queries_df.to_csv(QUERIES_CSV, index=False)

    time.sleep(2)
    clear_memory()

print("\nDone.", flush=True)
queries_df.to_csv(QUERIES_CSV, index=False)
print(f"Final annotated file saved to {QUERIES_CSV}", flush=True)