import os
import ast
import json
import warnings

import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

print("========================================")
print(" STAGE 4: STRATIFIED RETRIEVAL EVALUATION ")
print("========================================")

# =========================
# CONFIG
# =========================
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
BASE_PATH = os.path.join(project_root, "data", "processed")

QUERY_FILE = os.path.join(BASE_PATH, "queries_annotated.csv")
ID_MAPPING_FILE = os.path.join(BASE_PATH, "indices", "id_mapping.json")
INDICES_DIR = os.path.join(BASE_PATH, "indices")
OUTPUT_CSV = os.path.join(BASE_PATH, "detailed_evaluation_metrics.csv")
SUMMARY_CSV = os.path.join(BASE_PATH, "summary_evaluation_metrics.csv")

ALLOWED_LANGUAGES = {"English", "Tagalog", "Code-Switched"}
TOP_K = 10


def parse_list_value(value):
    """Safely parse a list-like value from CSV."""
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
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed if str(v).strip()]
            except Exception:
                try:
                    parsed = json.loads(text.replace("'", '"'))
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


def load_id_lookup(mapping_path):
    """
    Supports:
      1) list: [passage_id_0, passage_id_1, ...]
      2) dict index -> passage_id
      3) dict passage_id -> index (reversed automatically)
    """
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


def precision_at_k(retrieved_ids, ground_truth_ids, k):
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant = sum(1 for doc_id in top_k if doc_id in ground_truth_ids)
    return relevant / float(k)


def reciprocal_rank(retrieved_ids, ground_truth_ids):
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in ground_truth_ids:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved_ids, ground_truth_ids, k):
    if not ground_truth_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant = sum(1 for doc_id in top_k if doc_id in ground_truth_ids)
    return relevant / float(len(ground_truth_ids))


def main():
    print(f"Using BASE_PATH: {BASE_PATH}")
    print("Loading annotated queries...")

    queries_df = pd.read_csv(QUERY_FILE)

    required_columns = {
        "query_id",
        "query_text",
        "language_label",
        "semantic_type",
        "relevant_passage_ids",
    }
    missing_columns = required_columns - set(queries_df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in queries file: {sorted(missing_columns)}")

    queries_df["language_label"] = queries_df["language_label"].astype(str).str.strip()
    queries_df["semantic_type"] = queries_df["semantic_type"].astype(str).str.strip()
    queries_df["relevant_passage_ids"] = queries_df["relevant_passage_ids"].apply(parse_list_value)

    total_queries = len(queries_df)
    in_scope_df = queries_df[queries_df["language_label"].isin(ALLOWED_LANGUAGES)].copy()
    out_of_scope_df = queries_df[~queries_df["language_label"].isin(ALLOWED_LANGUAGES)].copy()
    judged_in_scope_df = in_scope_df[in_scope_df["relevant_passage_ids"].map(len) > 0].copy()

    print(f"Total queries in file: {total_queries}")
    print(f"In-scope queries (English/Tagalog/Code-Switched): {len(in_scope_df)}")
    print(f"Out-of-scope queries excluded: {len(out_of_scope_df)}")
    print(f"Judged in-scope queries used for evaluation: {len(judged_in_scope_df)}")

    if len(out_of_scope_df) > 0:
        print("\nExcluded out-of-scope queries:")
        print(out_of_scope_df[["query_id", "language_label", "semantic_type"]].to_string(index=False))

    if len(judged_in_scope_df) == 0:
        raise ValueError("No judged in-scope queries found. Cannot compute evaluation metrics.")

    id_lookup = load_id_lookup(ID_MAPPING_FILE)

    models = {
        "msbert": "paraphrase-multilingual-mpnet-base-v2",
        "legal_bert": "nlpaueb/legal-bert-base-uncased",
        "bge_m3": "BAAI/bge-m3",
    }

    all_query_metrics = []

    for model_alias, model_path in models.items():
        print(f"\nEvaluating {model_alias.upper()}...")

        try:
            encoder = SentenceTransformer(model_path, device="cpu")
            index_path = os.path.join(INDICES_DIR, f"{model_alias}_index.faiss")
            index = faiss.read_index(index_path)

            for _, row in judged_in_scope_df.iterrows():
                query_id = row["query_id"]
                query_text = row["query_text"]
                language = row["language_label"]
                semantic_type = row["semantic_type"]
                ground_truth_ids = unique_preserve_order(row["relevant_passage_ids"])

                query_vector = encoder.encode(
                    [query_text],
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                ).astype(np.float32)

                distances, faiss_indices = index.search(query_vector, TOP_K)

                retrieved_ids = []
                for faiss_id in faiss_indices[0]:
                    if faiss_id == -1:
                        continue
                    try:
                        retrieved_ids.append(str(id_lookup(int(faiss_id))))
                    except Exception as e:
                        warnings.warn(
                            f"Could not resolve FAISS index {faiss_id} for query {query_id} "
                            f"under model {model_alias.upper()}: {e}"
                        )

                mrr = reciprocal_rank(retrieved_ids, ground_truth_ids)
                p5 = precision_at_k(retrieved_ids, ground_truth_ids, 5)
                p10 = precision_at_k(retrieved_ids, ground_truth_ids, 10)
                recall10 = recall_at_k(retrieved_ids, ground_truth_ids, 10)

                all_query_metrics.append({
                    "Model": model_alias.upper(),
                    "Query_ID": query_id,
                    "Language": language,
                    "Semantic_Type": semantic_type,
                    "MRR": mrr,
                    "P@5": p5,
                    "P@10": p10,
                    "Recall@10": recall10,
                    "Retrieved_Ids": json.dumps(retrieved_ids, ensure_ascii=False),
                    "Ground_Truth_Ids": json.dumps(ground_truth_ids, ensure_ascii=False),
                })

            print(f"Finished {model_alias.upper()}.")

        except Exception as e:
            print(f"  > Skipping {model_alias.upper()} due to error: {e}")

    if not all_query_metrics:
        raise RuntimeError("No evaluation metrics were computed. Check model loading and index files.")

    metrics_df = pd.DataFrame(all_query_metrics)

    # Save query-level results
    metrics_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDetailed query-level metrics saved to: {OUTPUT_CSV}")

    print("\n" + "=" * 60)
    print(" GLOBAL OVERALL PERFORMANCE ")
    print("=" * 60)
    global_perf = metrics_df.groupby("Model")[["MRR", "P@5", "P@10", "Recall@10"]].mean().round(4)
    print(global_perf)

    print("\n" + "=" * 60)
    print(" STRATIFIED BY LANGUAGE ")
    print("=" * 60)
    lang_perf = metrics_df.groupby(["Model", "Language"])[["MRR", "P@5", "P@10", "Recall@10"]].mean().round(4)
    print(lang_perf)

    print("\n" + "=" * 60)
    print(" STRATIFIED BY SEMANTIC TYPE ")
    print("=" * 60)
    type_perf = metrics_df.groupby(["Model", "Semantic_Type"])[["MRR", "P@5", "P@10", "Recall@10"]].mean().round(4)
    print(type_perf)

    # Save summary tables in long format
    summary_rows = []

    for model_name, group in metrics_df.groupby("Model"):
        summary_rows.append({
            "Scope": "Overall",
            "Model": model_name,
            "Category": "All",
            "MRR": group["MRR"].mean(),
            "P@5": group["P@5"].mean(),
            "P@10": group["P@10"].mean(),
            "Recall@10": group["Recall@10"].mean(),
            "Num_Queries": len(group),
        })

    for (model_name, language), group in metrics_df.groupby(["Model", "Language"]):
        summary_rows.append({
            "Scope": "Language",
            "Model": model_name,
            "Category": language,
            "MRR": group["MRR"].mean(),
            "P@5": group["P@5"].mean(),
            "P@10": group["P@10"].mean(),
            "Recall@10": group["Recall@10"].mean(),
            "Num_Queries": len(group),
        })

    for (model_name, semantic_type), group in metrics_df.groupby(["Model", "Semantic_Type"]):
        summary_rows.append({
            "Scope": "Semantic_Type",
            "Model": model_name,
            "Category": semantic_type,
            "MRR": group["MRR"].mean(),
            "P@5": group["P@5"].mean(),
            "P@10": group["P@10"].mean(),
            "Recall@10": group["Recall@10"].mean(),
            "Num_Queries": len(group),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSummary metrics saved to: {SUMMARY_CSV}")

    print("\n" + "=" * 60)
    print(" COVERAGE REPORT ")
    print("=" * 60)
    print(f"Total queries in file: {total_queries}")
    print(f"In-scope queries: {len(in_scope_df)}")
    print(f"Out-of-scope queries excluded: {len(out_of_scope_df)}")
    print(f"Judged in-scope queries evaluated: {len(judged_in_scope_df)}")


if __name__ == "__main__":
    main()