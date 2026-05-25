import os
import ast
import json
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

print("========================================")
print("   STAGE 4: STRATIFIED RETRIEVAL EVALUATION ")
print("========================================")

# =========================
# CONFIG: Drive Path
# =========================
import os
# Get the absolute path to the project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
BASE_PATH = os.path.join(project_root, "data", "processed")

print(f"Using BASE_PATH: {BASE_PATH}")
print("Loading ground truth queries...")
queries_df = pd.read_csv(f"{BASE_PATH}/queries_annotated.csv")

with open(f"{BASE_PATH}/indices/id_mapping.json", "r") as f:
    id_mapping = json.load(f)

# Safely parse the string representation of lists back into actual Python lists
queries_df['relevant_passage_ids'] = queries_df['relevant_passage_ids'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else []
)

valid_queries = queries_df[queries_df['relevant_passage_ids'].map(len) > 0]
print(f"Evaluating on {len(valid_queries)} annotated queries out of {len(queries_df)} total.\n")

models = {
    "msbert": "paraphrase-multilingual-mpnet-base-v2",
    "legal_bert": "nlpaueb/legal-bert-base-uncased",
    "bge_m3": "BAAI/bge-m3"
}

all_query_metrics = []

for model_alias, model_path in models.items():
    print(f"Evaluating {model_alias.upper()}...")

    try:
        encoder = SentenceTransformer(model_path, device="cpu")
        index = faiss.read_index(f"{BASE_PATH}/indices/{model_alias}_index.faiss")

        for idx, row in valid_queries.iterrows():
            query_id = row['query_id']
            query_text = row['query_text']
            language = row['language_label']
            semantic_type = row['semantic_type']
            ground_truth_ids = row['relevant_passage_ids']

            query_vector = encoder.encode([query_text], normalize_embeddings=True)
            distances, faiss_indices = index.search(query_vector, 10)

            retrieved_ids = [str(id_mapping[int(faiss_id)]) for faiss_id in faiss_indices[0] if faiss_id != -1]

            # 1. Calculate MRR
            mrr = 0.0
            for rank, retrieved_id in enumerate(retrieved_ids, start=1):
                if retrieved_id in ground_truth_ids:
                    mrr = 1.0 / rank
                    break

            # 2. Calculate Precision@5
            top_5 = retrieved_ids[:5]
            relevant_in_top_5 = sum(1 for doc in top_5 if doc in ground_truth_ids)
            p5 = relevant_in_top_5 / 5.0

            # 3. Calculate Precision@10
            relevant_in_top_10 = sum(1 for doc in retrieved_ids if doc in ground_truth_ids)
            p10 = relevant_in_top_10 / 10.0

            # 4. Calculate Recall@10 (Added for Thesis strength)
            total_relevant = len(ground_truth_ids)
            recall_10 = relevant_in_top_10 / total_relevant if total_relevant > 0 else 0.0

            all_query_metrics.append({
                "Model": model_alias.upper(),
                "Query_ID": query_id,
                "Language": language,
                "Semantic_Type": semantic_type,
                "MRR": mrr,
                "P@5": p5,
                "P@10": p10,
                "Recall@10": recall_10
            })

    except Exception as e:
        print(f"  > Skipping {model_alias.upper()} (Error: {e})\n")

metrics_df = pd.DataFrame(all_query_metrics)

print("\n" + "="*60)
print("              GLOBAL OVERALL PERFORMANCE")
print("="*60)
global_perf = metrics_df.groupby("Model")[["MRR", "P@5", "P@10", "Recall@10"]].mean().round(4)
print(global_perf)

print("\n" + "="*60)
print("              STRATIFIED BY LANGUAGE")
print("="*60)
lang_perf = metrics_df.groupby(["Model", "Language"])[["MRR", "P@5", "P@10", "Recall@10"]].mean().round(4)
print(lang_perf)

print("\n" + "="*60)
print("            STRATIFIED BY SEMANTIC TYPE")
print("="*60)
type_perf = metrics_df.groupby(["Model", "Semantic_Type"])[["MRR", "P@5", "P@10", "Recall@10"]].mean().round(4)
print(type_perf)

# Save to Drive
output_csv = f"{BASE_PATH}/detailed_evaluation_metrics.csv"
metrics_df.to_csv(output_csv, index=False)
print(f"\nDetailed query-level metrics saved to '{output_csv}'")