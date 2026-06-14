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

# Exclude very short documents (< 50 words)
# Headers and stubs lack sufficient context for meaningful embeddings
df = df[df['text'].str.split().str.len() > 50]

print(f"Size after removing empty/short rows: {len(df)} rows")

# Define valid document categories for sampling
# Statutes: primary legislation (Republic Acts, Commonwealth Acts)
# Decisions: Supreme Court rulings
# Regulations: executive issuances (EOs, MCs, LOIs)
valid_statutes = ['Republic Acts', 'Commonwealth Act', 'Acts']
valid_decisions = ['Decisions / Signed Resolutions', 'Decisions / Sign Resolutions']
valid_regs = ['Executive Orders', 'Memorandum Circulars', 'Letter of Instruction']

statutes_df = df[df[CAT_COL].isin(valid_statutes)]
decisions_df = df[df[CAT_COL].isin(valid_decisions)]
regs_df = df[df[CAT_COL].isin(valid_regs)]

try:
    # Stratified sampling targets (2,400 total):
    # - 900 statutes (37.5%): Primary legislation for statutory interpretation queries
    # - 1,200 decisions (50%): Case law for precedent-based queries
    # - 300 regulations (12.5%): Executive issuances for regulatory queries
    # This distribution reflects typical legal research query patterns
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
