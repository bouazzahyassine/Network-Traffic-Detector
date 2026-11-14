import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

DATA_FILE = "data/processed/Tuesday_clean.csv"
PREPARED_FILE = "data/prepared/prepared_dataset.joblib"

print("=" * 70)
print("PHASE 14.4 - ORIGINAL FOLDS INTEGRITY AUDIT")
print("=" * 70)

df = pd.read_csv(DATA_FILE)
prepared = joblib.load(PREPARED_FILE)

indices = prepared["train_indices"]
y = np.asarray(prepared["y_train"])

X = df.iloc[indices].drop(columns=["Label"])
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)

# Original groups: all features, including Destination Port
groups_original = pd.util.hash_pandas_object(
    X, index=False
).to_numpy(dtype="uint64")

# Reduced groups: without Destination Port
X_without_port = X.drop(columns=["Destination Port"])

groups_without_port = pd.util.hash_pandas_object(
    X_without_port, index=False
).to_numpy(dtype="uint64")

print("Rows:", len(y))
print("Unique original groups:", len(np.unique(groups_original)))
print("Unique groups without port:", len(np.unique(groups_without_port)))

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\nChecking ORIGINAL Phase 14.4 folds...")
total_overlap = 0

for fold, (fit_idx, val_idx) in enumerate(
    cv.split(np.zeros(len(y)), y, groups_original), start=1
):
    fit_groups = np.unique(groups_without_port[fit_idx])
    val_groups = np.unique(groups_without_port[val_idx])

    overlap = np.intersect1d(fit_groups, val_groups)

    total_overlap += len(overlap)

    print(f"\nFold {fold}")
    print("Fit rows:", len(fit_idx))
    print("Validation rows:", len(val_idx))
    print("Overlapping reduced-feature groups:", len(overlap))

print("\n" + "=" * 70)
print("TOTAL OVERLAPPING GROUPS:", total_overlap)

if total_overlap == 0:
    print("RESULT: ORIGINAL FOLDS PASSED")
    print("Phase 14.4 grouping is safe after removing Destination Port.")
else:
    print("RESULT: ORIGINAL FOLDS FAILED")
    print("Rerun both models using groups_without_port.")
print("=" * 70)
