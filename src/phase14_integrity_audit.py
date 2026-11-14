#!/usr/bin/env python3

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

print("=" * 70)
print("PHASE 14.1 - DATASET INTEGRITY AND DUPLICATE LEAKAGE AUDIT")
print("=" * 70)

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

DATA_PATH = "data/processed/Tuesday_clean.csv"
PREPARED_PATH = "data/prepared/prepared_dataset.joblib"
OUTPUT_DIR = "results/robustness"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("\n[1] Loading original processed dataset...")

df = pd.read_csv(DATA_PATH, low_memory=False)
prepared = joblib.load(PREPARED_PATH)

print("Original dataset:", df.shape)

X_train = prepared["X_train"]
X_test = prepared["X_test"]
y_train = np.asarray(prepared["y_train"])
y_test = np.asarray(prepared["y_test"])

train_indices = np.asarray(prepared["train_indices"])
test_indices = np.asarray(prepared["test_indices"])

# --------------------------------------------------
# 3. BASIC INTEGRITY CHECKS
# --------------------------------------------------

print("\n[2] Checking dataset integrity...")

checks = {}

checks["original_rows"] = int(len(df))
checks["train_rows"] = int(len(X_train))
checks["test_rows"] = int(len(X_test))
checks["train_test_rows_match"] = (
    len(X_train) == len(y_train)
    and len(X_test) == len(y_test)
)

checks["feature_dimensions_match"] = (
    X_train.shape[1] == X_test.shape[1]
    == len(prepared["feature_names"])
)

checks["train_indices_unique"] = (
    len(np.unique(train_indices)) == len(train_indices)
)

checks["test_indices_unique"] = (
    len(np.unique(test_indices)) == len(test_indices)
)

checks["train_test_indices_disjoint"] = (
    len(np.intersect1d(train_indices, test_indices)) == 0
)

checks["indices_cover_dataset"] = (
    len(np.union1d(train_indices, test_indices)) == len(df)
)

checks["train_finite_values"] = bool(np.isfinite(X_train).all())
checks["test_finite_values"] = bool(np.isfinite(X_test).all())

for name, result in checks.items():
    print(f"{name}: {result}")

# --------------------------------------------------
# 4. RECREATE ORIGINAL DUPLICATE GROUPS
# --------------------------------------------------

print("\n[3] Recreating duplicate groups...")

if "Label" not in df.columns:
    raise ValueError("Label column not found.")

labels_original = df["Label"].astype(str).str.strip()

X_original = df.drop(columns=["Label"]).copy()

for col in X_original.columns:
    X_original[col] = pd.to_numeric(
        X_original[col],
        errors="coerce"
    )

X_original = X_original.replace([np.inf, -np.inf], np.nan)

groups = pd.util.hash_pandas_object(
    X_original,
    index=False
).to_numpy(dtype="uint64")

print("Total rows:", len(groups))
print("Unique feature groups:", len(np.unique(groups)))
print("Repeated rows:", len(groups) - len(np.unique(groups)))

# --------------------------------------------------
# 5. VERIFY PHASE 10 TRAIN/TEST GROUP SEPARATION
# --------------------------------------------------

print("\n[4] Checking Phase 10 train/test group separation...")

groups_train = groups[train_indices]
groups_test = groups[test_indices]

phase10_overlap = np.intersect1d(
    np.unique(groups_train),
    np.unique(groups_test)
)

print("Overlapping groups:", len(phase10_overlap))

# --------------------------------------------------
# 6. REPRODUCE PHASE 12 INTERNAL VALIDATION SPLIT
# --------------------------------------------------

print("\n[5] Reproducing Phase 12 internal split...")

y_train_binary = np.where(y_train == "BENIGN", 0, 1)

fit_positions, val_positions = train_test_split(
    np.arange(len(y_train)),
    test_size=0.20,
    random_state=42,
    stratify=y_train_binary
)

groups_fit = groups_train[fit_positions]
groups_val = groups_train[val_positions]

internal_overlap = np.intersect1d(
    np.unique(groups_fit),
    np.unique(groups_val)
)

print("Internal training rows:", len(fit_positions))
print("Internal validation rows:", len(val_positions))
print("Unique groups in fit:", len(np.unique(groups_fit)))
print("Unique groups in validation:", len(np.unique(groups_val)))
print("Overlapping internal groups:", len(internal_overlap))

# --------------------------------------------------
# 7. MEASURE OVERLAPPING ROWS
# --------------------------------------------------

print("\n[6] Measuring duplicate overlap...")

overlap_set = set(internal_overlap.tolist())

fit_overlap_mask = np.isin(groups_fit, internal_overlap)
val_overlap_mask = np.isin(groups_val, internal_overlap)

fit_overlap_rows = int(fit_overlap_mask.sum())
val_overlap_rows = int(val_overlap_mask.sum())

print("Fit rows belonging to overlapping groups:", fit_overlap_rows)
print("Validation rows belonging to overlapping groups:", val_overlap_rows)

# Build group-level details
if len(internal_overlap) > 0:
    fit_counts = pd.Series(groups_fit).value_counts()
    val_counts = pd.Series(groups_val).value_counts()

    overlap_df = pd.DataFrame({
        "group_id": internal_overlap,
        "fit_rows": [
            int(fit_counts.get(g, 0)) for g in internal_overlap
        ],
        "validation_rows": [
            int(val_counts.get(g, 0)) for g in internal_overlap
        ]
    })

    overlap_df.to_csv(
        f"{OUTPUT_DIR}/phase14_internal_duplicate_overlap.csv",
        index=False
    )
else:
    overlap_df = pd.DataFrame(
        columns=["group_id", "fit_rows", "validation_rows"]
    )

# --------------------------------------------------
# 8. CHECK GROUPS WITH CONFLICTING LABELS
# --------------------------------------------------

print("\n[7] Checking conflicting labels inside duplicate groups...")

group_label_counts = pd.DataFrame({
    "group_id": groups,
    "label": labels_original.to_numpy()
}).groupby("group_id")["label"].nunique()

conflicting_groups = int((group_label_counts > 1).sum())

print("Groups with conflicting labels:", conflicting_groups)

# --------------------------------------------------
# 9. SUMMARY
# --------------------------------------------------

summary = {
    "phase": "14.1",
    "title": "Dataset Integrity and Duplicate Leakage Audit",
    "original_rows": int(len(df)),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "features": int(X_train.shape[1]),
    "unique_feature_groups": int(len(np.unique(groups))),
    "repeated_feature_rows": int(
        len(groups) - len(np.unique(groups))
    ),
    "phase10_train_test_overlapping_groups": int(
        len(phase10_overlap)
    ),
    "phase12_internal_fit_rows": int(len(fit_positions)),
    "phase12_internal_validation_rows": int(len(val_positions)),
    "phase12_internal_overlapping_groups": int(
        len(internal_overlap)
    ),
    "fit_rows_in_overlapping_groups": fit_overlap_rows,
    "validation_rows_in_overlapping_groups": val_overlap_rows,
    "conflicting_label_groups": conflicting_groups,
    "integrity_checks": checks
}

with open(
    f"{OUTPUT_DIR}/phase14_integrity_summary.json",
    "w"
) as f:
    json.dump(summary, f, indent=4)

print("\n" + "=" * 70)
print("PHASE 14.1 COMPLETED")
print("=" * 70)

print("\nSummary:")
print(json.dumps(summary, indent=4))

print("\nSaved:")
print(f"{OUTPUT_DIR}/phase14_integrity_summary.json")

if len(internal_overlap) > 0:
    print(
        f"{OUTPUT_DIR}/phase14_internal_duplicate_overlap.csv"
    )
