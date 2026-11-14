import gc
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold

print("=" * 70)
print("PHASE 14.4 - INTEGRITY AUDIT WITHOUT DESTINATION PORT")
print("=" * 70)

df = pd.read_csv("data/processed/Tuesday_clean.csv")
prepared = joblib.load("data/prepared/prepared_dataset.joblib")

indices = np.asarray(prepared["train_indices"])
y = np.asarray(prepared["y_train"]).astype(str)

del prepared
gc.collect()

train_df = df.iloc[indices]

if not np.array_equal(
    train_df["Label"].astype(str).to_numpy(), y
):
    raise ValueError("Label mismatch!")

X = train_df.drop(columns=["Label"])

for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

X = X.replace([np.inf, -np.inf], np.nan)

del df, train_df, indices
gc.collect()

print("\nOriginal rows:", len(X))
print("Original features:", X.shape[1])

# Remove Destination Port BEFORE reconstructing groups
X_no_port = X.drop(columns=["Destination Port"])

groups_no_port = pd.util.hash_pandas_object(
    X_no_port,
    index=False
).to_numpy(dtype="uint64")

print("Features without port:", X_no_port.shape[1])
print("Unique groups without port:",
      len(np.unique(groups_no_port)))

del X, X_no_port
gc.collect()

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\nChecking all five folds...")

total_overlap = 0

for fold, (fit_idx, val_idx) in enumerate(
    cv.split(np.zeros(len(y)), y, groups_no_port),
    start=1
):
    fit_groups = np.unique(groups_no_port[fit_idx])
    val_groups = np.unique(groups_no_port[val_idx])

    overlap = np.intersect1d(
        fit_groups,
        val_groups
    )

    print(f"\nFold {fold}")
    print("Fit rows:", len(fit_idx))
    print("Validation rows:", len(val_idx))
    print("Overlapping groups:", len(overlap))

    total_overlap += len(overlap)

    if len(overlap):
        print("WARNING: GROUP LEAKAGE DETECTED")

    del fit_groups, val_groups, overlap

print("\n" + "=" * 70)
print("INTEGRITY AUDIT SUMMARY")
print("=" * 70)
print("Total overlapping groups:", total_overlap)

if total_overlap == 0:
    print("RESULT: PASSED")
else:
    print("RESULT: FAILED - STRICTER SPLIT REQUIRED")

del groups_no_port, y
gc.collect()
