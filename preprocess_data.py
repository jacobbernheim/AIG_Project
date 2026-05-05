import pandas as pd

# ── Base paths ───────────────────────────────────────────────────────────────
RAW_DIR       = '/Users/jacobbernheim/Github-Projects/AIG_Project/data/raw/'
PROCESSED_DIR = '/Users/jacobbernheim/Github-Projects/AIG_Project/data/processed/'

# ── Load files ───────────────────────────────────────────────────────────────
activities_df = pd.read_csv(RAW_DIR + 'Payload Activities.csv')
categories_df = pd.read_csv(RAW_DIR + 'Payload Categories.csv')
sequences_df  = pd.read_csv(RAW_DIR + 'all_sequences.csv', header=None)

# ── Clean up sequences file ───────────────────────────────────────────────────
sequences_df.columns = ['MenDel.Name', 'Sequence']
sequences_df['MenDel.Name'] = sequences_df['MenDel.Name'].str.strip()

# ── Strip whitespace from MenDel.Name across all dfs ─────────────────────────
for df in [activities_df, categories_df]:
    df['MenDel.Name'] = df['MenDel.Name'].str.strip()

# ── Average numeric replicate columns in Activities ───────────────────────────
cols_to_avg  = ['Sox2 (CAST)', 'Sox2 (BL6)', 'Fold Change', 'Activity']
cols_to_keep = ['Project', 'groups', 'PL', 'MenDel.Name', 'Paper']

# Keep one representative row for the non-numeric columns
meta_df = (
    activities_df[cols_to_keep]
    .groupby('MenDel.Name', as_index=False)
    .first()
)

# Average the numeric columns across replicates
avg_df = (
    activities_df[['MenDel.Name'] + cols_to_avg]
    .groupby('MenDel.Name', as_index=False)
    .mean(numeric_only=True)
)

activities_collapsed = meta_df.merge(avg_df, on='MenDel.Name')

# ── Merge everything together ─────────────────────────────────────────────────
# 1. Activities (collapsed) + Categories on MenDel.Name
merged = activities_collapsed.merge(
    categories_df[['MenDel.Name', 'Category']],
    on='MenDel.Name',
    how='outer'
)

# 2. Add sequences
merged = merged.merge(
    sequences_df,
    on='MenDel.Name',
    how='outer'
)

# ── Final column order ────────────────────────────────────────────────────────
final_cols = [
    'MenDel.Name',
    'Project',
    'groups',
    'PL',
    'Category',
    'Paper',
    'Sox2 (CAST)',
    'Sox2 (BL6)',
    'Fold Change',
    'Activity',
    'Sequence'
]

# Only include columns that actually exist
final_cols = [c for c in final_cols if c in merged.columns]
merged = merged[final_cols]

# ── Save output ───────────────────────────────────────────────────────────────
output_path = PROCESSED_DIR + 'merged_payloads.csv'
merged.to_csv(output_path, index=False)

print(f"Done! Output saved to: {output_path}")
print(f"Shape: {merged.shape[0]} rows × {merged.shape[1]} columns")
print(f"\nColumn names:\n{list(merged.columns)}")

# ── Sanity check ──────────────────────────────────────────────────────────────
print(f"\n── Row counts ──────────────────────────────────────────────────")
print(f"Unique MenDel.Names in Activities:  {activities_df['MenDel.Name'].nunique()}")
print(f"Unique MenDel.Names in Categories:  {categories_df['MenDel.Name'].nunique()}")
print(f"Unique MenDel.Names in Sequences:   {sequences_df['MenDel.Name'].nunique()}")
print(f"Rows in final merged file:          {merged.shape[0]}")