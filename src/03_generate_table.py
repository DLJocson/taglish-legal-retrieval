"""Generate Table 1 corpus statistics for the PLRB processed passages.

Aggregates passage counts, language distribution, and mean token length by
broad legal document category and exports
``data/processed/Table_1_Corpus_Statistics.csv`` for thesis reporting.
"""

import pandas as pd

print("Generating Table 1 Statistics from processed data...")

df = pd.read_csv("data/processed/module1_processed_passages.csv")

category_mapping = {
    'Republic Acts': 'Republic Acts / Pres. Decrees',
    'Acts': 'Republic Acts / Pres. Decrees',
    'Commonwealth Act': 'Republic Acts / Pres. Decrees',
    'Decisions / Signed Resolutions': 'Supreme Court Decisions',
    'Executive Orders': 'Administrative Regulations',
    'Memorandum Circulars': 'Administrative Regulations',
    'Letter of Instruction': 'Administrative Regulations'
}
df['Broad_Category'] = df['document_type'].map(category_mapping)

df['Tokens'] = df['passage_text'].astype(str).apply(lambda x: len(x.split()))

lang_counts = pd.crosstab(df['Broad_Category'], df['language'])

for col in ['English', 'Filipino', 'Code-Switched']:
    if col not in lang_counts.columns:
        lang_counts[col] = 0

agg_df = df.groupby('Broad_Category').agg(
    Total=('passage_id', 'count'),
    Avg_Tokens=('Tokens', 'mean')
)

table = agg_df.join(lang_counts)

table = table[['Total', 'English', 'Filipino', 'Code-Switched', 'Avg_Tokens']]

row_order = ['Republic Acts / Pres. Decrees', 'Supreme Court Decisions', 'Administrative Regulations']
table = table.reindex(row_order)

total_row = pd.DataFrame({
    'Total': [table['Total'].sum()],
    'English': [table['English'].sum()],
    'Filipino': [table['Filipino'].sum()],
    'Code-Switched': [table['Code-Switched'].sum()],
    'Avg_Tokens': [df['Tokens'].mean()]
}, index=['Total'])

final_table = pd.concat([table, total_row])

final_table['Avg_Tokens'] = final_table['Avg_Tokens'].round().astype(int)

output_path = "data/processed/Table_1_Corpus_Statistics.csv"
final_table.to_csv(output_path, index_label="Document Type")

print("\n" + "="*80)
print("   TABLE 1: PLRB CORPUS STATISTICS (ACTUAL EMPIRICAL DATA)")
print("="*80)
print(final_table.to_string())
print("="*80)
print(f"\nDone! You can open '{output_path}' in Excel and copy it directly into your Word document.")
