"""Download, filter, and stratified-sample the OmniCorpus Philippine legal corpus.

Authenticates to Hugging Face, loads the philippine_laws parquet segment, applies
quality filters, samples 2,400 documents across statutes, decisions, and
regulations, and exports ``data/raw/master_legal_corpus.csv`` for preprocessing.
"""

import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

print("Downloading and cleaning laws segment from OmniCorpus...")

path = "hf://datasets/mongramosjr/philippine-omnicorpus/data/philippine_laws.parquet"
df = pd.read_parquet(
    path,
    storage_options={"token": hf_token},
)

CAT_COL = 'label'

print(f"Original dataset size: {len(df)} rows")

df = df.dropna(subset=['text', CAT_COL])

# Exclude very short documents; headers and stubs lack embedding-ready context.
df = df[df['text'].str.split().str.len() > 50]

print(f"Size after removing empty/short rows: {len(df)} rows")

valid_statutes = ['Republic Acts', 'Commonwealth Act', 'Acts']
valid_decisions = ['Decisions / Signed Resolutions', 'Decisions / Sign Resolutions']
valid_regs = ['Executive Orders', 'Memorandum Circulars', 'Letter of Instruction']

statutes_df = df[df[CAT_COL].isin(valid_statutes)]
decisions_df = df[df[CAT_COL].isin(valid_decisions)]
regs_df = df[df[CAT_COL].isin(valid_regs)]

try:
    # Target: 900 statutes, 1,200 decisions, 300 regulations (2,400 total).
    statutes = statutes_df.sample(n=900, random_state=42)
    decisions = decisions_df.sample(n=1200, random_state=42)
    regs = regs_df.sample(n=300, random_state=42)

    plrb_corpus = pd.concat([statutes, decisions, regs])

    os.makedirs("data/raw", exist_ok=True)

    plrb_corpus.to_csv("data/raw/master_legal_corpus.csv", index=False)

    print("\n--- Cleaning and Extraction Complete ---")
    print(f"Successfully saved 2,400 strictly validated documents to data/raw/master_legal_corpus.csv")
    print("Document Breakdown:")
    print(plrb_corpus[CAT_COL].value_counts())

except ValueError as e:
    print(f"\nError during sampling: {e}")
    print(f"Available Statutes: {len(statutes_df)}")
    print(f"Available Decisions: {len(decisions_df)}")
    print(f"Available Regulations: {len(regs_df)}")
