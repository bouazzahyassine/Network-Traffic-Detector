
import pandas as pd
import os

FILE = "data/processed/Tuesday_clean.csv"
OUTPUT = "results/eda"

print("=" * 55)
print("PHASE 9 - GLOBAL DUPLICATE AUDIT")
print("=" * 55)

print("\n[1] Loading dataset...")
df = pd.read_csv(FILE, low_memory=False)
df.columns = df.columns.str.strip()

print("Dataset shape:", df.shape)

print("\n[2] Checking exact duplicate rows...")
exact_mask = df.duplicated(keep="first")
print("Exact duplicate rows:", exact_mask.sum())

print("\nDuplicates by class:")
print(df.loc[exact_mask, "Label"].value_counts())

print("\n[3] Checking duplicate feature vectors...")
features = df.drop(columns=["Label"])

feature_hash = pd.util.hash_pandas_object(
    features, index=False
)

audit = pd.DataFrame({
    "hash": feature_hash,
    "Label": df["Label"]
})

repeated = audit.groupby("hash").agg(
    rows=("Label", "size"),
    labels=("Label", "nunique")
)

conflicting_hashes = repeated[
    repeated["labels"] > 1
].index

conflict_rows = audit[
    audit["hash"].isin(conflicting_hashes)
]

print("Repeated feature groups:", (repeated["rows"] > 1).sum())
print("Conflicting feature groups:", len(conflicting_hashes))
print("Rows in conflicting groups:", len(conflict_rows))

print("\n[4] Feature duplicate statistics...")
print("Repeated feature rows:", features.duplicated().sum())

os.makedirs(OUTPUT, exist_ok=True)

df.loc[exact_mask].to_csv(
    f"{OUTPUT}/exact_duplicate_rows.csv",
    index=False
)

conflict_rows.to_csv(
    f"{OUTPUT}/conflicting_labels.csv",
    index=False
)

with open(f"{OUTPUT}/global_duplicate_summary.txt", "w") as f:
    f.write(f"Total rows: {len(df)}\n")
    f.write(f"Exact duplicates: {exact_mask.sum()}\n")
    f.write(f"Repeated feature groups: {(repeated['rows'] > 1).sum()}\n")
    f.write(f"Conflicting feature groups: {len(conflicting_hashes)}\n")
    f.write(f"Rows in conflicting groups: {len(conflict_rows)}\n")

print("\nResults saved successfully!")
print("PHASE 9 GLOBAL DUPLICATE AUDIT COMPLETED!")
