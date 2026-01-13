import pandas as pd
import os

file_path = r'C:\Users\ilove\Desktop\解析\jyogai20251223_duo2_azukun.csv'

print(f"Reading {file_path}...")
df = pd.read_csv(file_path, encoding='cp932')

# Ensure Dedama is numeric
df['Dedama'] = pd.to_numeric(df['Dedama'], errors='coerce').fillna(0).astype(int)

# 1. Identify rows with Dedama >= 230
high_dedama_rows = df[df['Dedama'] >= 230]
print(f"Found {len(high_dedama_rows)} rows with Dedama >= 230.")

# 2. Get the unique IDs from these rows
bad_ids = high_dedama_rows['ID'].unique()
print(f"Found {len(bad_ids)} unique IDs containing Dedama >= 230.")

# 3. Filter the original dataframe to exclude these IDs
original_count = len(df)
df_filtered = df[~df['ID'].isin(bad_ids)]
filtered_count = len(df_filtered)

print(f"Original row count: {original_count}")
print(f"Filtered row count: {filtered_count}")
print(f"Removed rows: {original_count - filtered_count}")

# 4. Save back to the file
df_filtered.to_csv(file_path, index=False)
print("File overwritten with filtered data.")
