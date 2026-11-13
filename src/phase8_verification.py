
import pandas as pd
import numpy as np
from collections import Counter

FILE = "data/processed/Tuesday_clean.csv"
CHUNK_SIZE = 50000

total_rows = 0
labels = Counter()
missing = Counter()
infinite = Counter()
duplicates = 0
first_chunk = True

print("Starting complete dataset verification...")

for chunk in pd.read_csv(
    FILE,
    chunksize=CHUNK_SIZE,
    low_memory=False
):

    chunk.columns = chunk.columns.str.strip()

    total_rows += len(chunk)

    # Analyse des classes
    labels.update(
        chunk["Label"].astype(str).str.strip().value_counts().to_dict()
    )

    # Analyse des caractéristiques numériques
    features = [c for c in chunk.columns if c != "Label"]

    for col in features:
        missing[col] += chunk[col].isna().sum()

        values = pd.to_numeric(chunk[col], errors="coerce")
        infinite[col] += np.isinf(values).sum()

    # Vérification des doublons à l'intérieur de chaque bloc
    duplicates += chunk.duplicated().sum()

    if first_chunk:
        columns = chunk.columns.tolist()
        dtypes = chunk.dtypes.astype(str).to_dict()
        first_chunk = False

print("\n========== DATASET SUMMARY ==========")
print("Total rows:", total_rows)
print("Total columns:", len(columns))
print("Total features:", len(columns) - 1)
print("Number of classes:", len(labels))

print("\n========== LABEL DISTRIBUTION ==========")
for label, count in labels.most_common():
    print(f"{label}: {count} ({count / total_rows * 100:.2f}%)")

print("\n========== MISSING VALUES ==========")
missing_found = {k: v for k, v in missing.items() if v > 0}
print(missing_found if missing_found else "No missing values")

print("\n========== INFINITE VALUES ==========")
infinite_found = {k: v for k, v in infinite.items() if v > 0}
print(infinite_found if infinite_found else "No infinite values")

print("\n========== DATA TYPES ==========")
for col, dtype in dtypes.items():
    print(f"{col}: {dtype}")

print("\n========== DUPLICATES ==========")
print("Duplicates found inside chunks:", duplicates)

print("\nVerification completed!")
