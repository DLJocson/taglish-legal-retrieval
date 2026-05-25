import os
import gc
import json
import shutil
import time
import pandas as pd
import faiss

from itertools import zip_longest
from google.colab import auth
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer

# =========================
# AUTH / ENV
# =========================
auth.authenticate_user()
os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)

# =========================
# CONFIG
# =========================
MODEL_ID = "gemini-2.5-flash"

PROJECT_ID = "project-fc4bdbdc-0585-464a-864"
LOCATION = "us-central1"

BASE_PATH = "/content/drive/MyDrive/Colab Notebooks/05_llm_annotator/comparison-of-embedding-models-ph-legal-text/data/processed"

QUERIES_CSV = f"{BASE_PATH}/queries.csv"
QUERIES_BACKUP_CSV = f"{BASE_PATH}/queries_original.csv"
PASSAGES_CSV = f"{BASE_PATH}/module1_processed_passages.csv"
ID_MAPPING_JSON = f"{BASE_PATH}/indices/id_mapping.json"

START_FROM_INDEX = 0
TOP_K_PER_MODEL = 5
MAX_PASSAGES_IN_PROMPT = 12
MAX_CHARS_PER_PASSAGE = 1200
SAVE_EVERY = 1
MAX_RETRIES = 3

MODEL_CONFIGS = {
    "msbert": "paraphrase-multilingual-mpnet-base-v2",
    "bge_m3": "BAAI/bge-m3",
    "legal_bert": "nlpaueb/legal-bert-base-uncased"
}

# =========================
# HELPERS
# =========================
def clear_memory():
    gc.collect()

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
- Only include IDs from retrieved passages
- Include ALL relevant passages (maximize recall)
- If none are relevant, return: {{"passage_ids": []}}
- DO NOT explain or chat
- Output MUST be raw JSON only
""".strip()

def safe_json_parse(text):
    try:
        text = (text or "").strip()
        if "```" in text:
            text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception:
        return {"passage_ids": []}

# =========================
# GEMINI CLIENT
# =========================
client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
)

# =========================
# LOAD DATA
# =========================
print("Loading dataset...", flush=True)

if os.path.exists(QUERIES_BACKUP_CSV) and START_FROM_INDEX == 0:
    print("Restoring original dataset...", flush=True)
    shutil.copyfile(QUERIES_BACKUP_CSV, QUERIES_CSV)

queries_df = pd.read_csv(QUERIES_CSV)
passages_df = pd.read_csv(PASSAGES_CSV)

# --- PANDAS WARNING FIX ---
queries_df["relevant_passage_ids"] = queries_df["relevant_passage_ids"].astype("object")

if "retrieval_sources" not in queries_df.columns:
    queries_df["retrieval_sources"] = pd.Series(
        [""] * len(queries_df),
        dtype="object"
    )
else:
    queries_df["retrieval_sources"] = queries_df["retrieval_sources"].astype("object")
# --------------------------

passages_df["passage_id"] = passages_df["passage_id"].astype(str)

with open(ID_MAPPING_JSON, "r") as f:
    id_mapping = json.load(f)

passage_lookup = passages_df.set_index("passage_id")["passage_text"].to_dict()

# =========================
# LOAD RETRIEVERS
# =========================
print("Loading retrieval models (CPU)...", flush=True)

loaded_systems = {}
for alias, path in MODEL_CONFIGS.items():
    encoder = SentenceTransformer(path, device="cpu")
    index_path = f"{BASE_PATH}/indices/{alias}_index.faiss"
    index = faiss.read_index(index_path)
    loaded_systems[alias] = {"encoder": encoder, "index": index}

# =========================
# ANNOTATION LOOP
# =========================
for idx, row in queries_df.iloc[START_FROM_INDEX:].iterrows():
    existing = str(row["relevant_passage_ids"])
    if pd.notna(row["relevant_passage_ids"]) and "[" in existing:
        print(f"Skipping {row['query_id']} already processed.", flush=True)
        continue

    query_text = str(row["query_text"])
    print(f"\n[{idx+1}/{len(queries_df)}] Processing {row['query_id']}...", flush=True)

    # Retrieval
    results_per_model = []
    provenance = {}

    for alias, system in loaded_systems.items():
        print(f"  Retrieving with {alias}...", flush=True)
        query_vec = system["encoder"].encode([query_text], normalize_embeddings=True)
        _, indices = system["index"].search(query_vec, TOP_K_PER_MODEL)
        results_per_model.append(indices[0])

        for fid in indices[0]:
            if fid != -1 and 0 <= fid < len(id_mapping):
                pid = str(id_mapping[fid])
                provenance.setdefault(pid, set()).add(alias)

    # Interleaving
    pooled_ids = []
    seen = set()

    for group in zip_longest(*results_per_model):
        for fid in group:
            if fid is None:
                continue
            if fid != -1 and 0 <= fid < len(id_mapping):
                pid = str(id_mapping[fid])
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

    for attempt in range(MAX_RETRIES):
        try:
            print(f"  Gemini attempt {attempt + 1}...", flush=True)
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0
                )
            )

            parsed = safe_json_parse(response.text)
            relevant_ids = [pid for pid in parsed.get("passage_ids", []) if pid in pooled_ids]
            success = True
            print(f"  Success: {relevant_ids}", flush=True)
            break

        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {type(e).__name__}: {e}", flush=True)
            time.sleep((attempt + 1) * 5)

    if success:
        queries_df.at[idx, "relevant_passage_ids"] = str(relevant_ids)
        queries_df.at[idx, "retrieval_sources"] = str({
            pid: list(provenance.get(pid, [])) for pid in relevant_ids
        })
        print("  Saved row.", flush=True)

    if SAVE_EVERY == 1:
        queries_df.to_csv(QUERIES_CSV, index=False)

    time.sleep(10)
    clear_memory()

print("\nDone.", flush=True)