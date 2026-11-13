
#!/usr/bin/env python3
"""
PHASE 10 - DATA PREPARATION FOR MACHINE LEARNING

Dataset: CIC-IDS2017 Tuesday
Strategy:
- Group-aware stratified train/test split
- Prevent repeated feature vectors from crossing splits
- Imputation fitted on training data only
- Constant feature removal fitted on training data only
- Save prepared datasets and preprocessing pipeline
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")

# =====================================================
# 1. PATHS AND CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data/processed/Tuesday_clean.csv"
OUTPUT_DIR = ROOT / "data/prepared"
RESULTS_DIR = ROOT / "results/preparation"
MODELS_DIR = ROOT / "models"

for directory in [OUTPUT_DIR, RESULTS_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

print("=" * 65)
print("PHASE 10 - DATA PREPARATION FOR MACHINE LEARNING")
print("=" * 65)

# =====================================================
# 2. LOAD DATASET
# =====================================================

print("\n[1] Loading dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, low_memory=False)

print("Dataset shape:", df.shape)

if "Label" not in df.columns:
    raise ValueError("Target column 'Label' not found.")

# =====================================================
# 3. SEPARATE FEATURES AND TARGET
# =====================================================

print("\n[2] Separating features and target...")

y = df["Label"].astype(str).str.strip()
X = df.drop(columns=["Label"]).copy()

if y.isna().any() or (y == "").any():
    raise ValueError("Missing or empty target labels detected.")

# Convert features to numeric
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# Replace infinite values
X = X.replace([np.inf, -np.inf], np.nan)

print("Number of features:", X.shape[1])
print("\nClass distribution:")
print(y.value_counts())

# =====================================================
# 4. CREATE GROUPS TO PREVENT DUPLICATE LEAKAGE
# =====================================================

print("\n[3] Creating duplicate groups...")

# Identical feature vectors receive the same group ID.
# This prevents repeated vectors from appearing in both
# training and testing sets.

groups = pd.util.hash_pandas_object(
    X,
    index=False
).to_numpy(dtype="uint64")

print("Total unique feature groups:", len(np.unique(groups)))
print("Repeated feature rows:", len(groups) - len(np.unique(groups)))

# =====================================================
# 5. GROUP-AWARE STRATIFIED SPLIT
# =====================================================

print("\n[4] Performing group-aware stratified split...")

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()

y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()

groups_train = groups[train_idx]
groups_test = groups[test_idx]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

print(
    "Training percentage:",
    round(len(X_train) / len(X) * 100, 2), "%"
)

print(
    "Testing percentage:",
    round(len(X_test) / len(X) * 100, 2), "%"
)

# =====================================================
# 6. VERIFY DUPLICATE GROUP SEPARATION
# =====================================================

print("\n[5] Checking duplicate leakage...")

overlap = np.intersect1d(
    np.unique(groups_train),
    np.unique(groups_test)
)

print("Overlapping feature groups:", len(overlap))

if len(overlap) > 0:
    raise RuntimeError(
        "Data leakage detected: repeated feature groups "
        "exist in both training and testing."
    )

print("Duplicate leakage check: PASSED")

# =====================================================
# 7. CLASS DISTRIBUTION
# =====================================================

print("\n[6] Checking class distribution...")

train_distribution = y_train.value_counts()
test_distribution = y_test.value_counts()

distribution = pd.DataFrame({
    "Training": train_distribution,
    "Testing": test_distribution
}).fillna(0).astype(int)

distribution["Total"] = (
    distribution["Training"] + distribution["Testing"]
)

distribution.to_csv(
    RESULTS_DIR / "class_distribution_split.csv"
)

print(distribution)

# =====================================================
# 8. PREPROCESSING PIPELINE
# =====================================================

print("\n[7] Building preprocessing pipeline...")

pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "feature_selector",
        VarianceThreshold(threshold=0.0)
    )
])

# Fit ONLY on training data
X_train_prepared = pipeline.fit_transform(X_train)

# Transform test data using the fitted training pipeline
X_test_prepared = pipeline.transform(X_test)

# Reduce memory usage
X_train_prepared = X_train_prepared.astype(np.float32)
X_test_prepared = X_test_prepared.astype(np.float32)

selected_features = X.columns[
    pipeline.named_steps["feature_selector"].get_support()
].tolist()

removed_features = [
    col for col in X.columns
    if col not in selected_features
]

print("Original features:", X.shape[1])
print("Selected features:", len(selected_features))
print("Removed constant features:", len(removed_features))

# =====================================================
# 9. SAVE PREPROCESSING PIPELINE
# =====================================================

print("\n[8] Saving preprocessing pipeline...")

joblib.dump(
    pipeline,
    MODELS_DIR / "preprocessing_pipeline.pkl"
)

# =====================================================
# 10. SAVE PREPARED DATA
# =====================================================

print("\n[9] Saving prepared datasets...")

prepared_data = {
    "X_train": X_train_prepared,
    "X_test": X_test_prepared,
    "y_train": y_train.to_numpy(),
    "y_test": y_test.to_numpy(),
    "feature_names": selected_features,
    "train_indices": train_idx,
    "test_indices": test_idx,
    "random_state": RANDOM_STATE
}

joblib.dump(
    prepared_data,
    OUTPUT_DIR / "prepared_dataset.joblib",
    compress=3
)

# Save feature selection report
pd.DataFrame({
    "Selected_Feature": selected_features
}).to_csv(
    RESULTS_DIR / "selected_features.csv",
    index=False
)

pd.DataFrame({
    "Removed_Constant_Feature": removed_features
}).to_csv(
    RESULTS_DIR / "removed_constant_features.csv",
    index=False
)

# =====================================================
# 11. SAVE SUMMARY
# =====================================================

summary = {
    "phase": 10,
    "dataset": "CIC-IDS2017 Tuesday",
    "original_rows": int(len(df)),
    "original_features": int(X.shape[1]),
    "training_rows": int(len(X_train)),
    "testing_rows": int(len(X_test)),
    "training_features": int(X_train_prepared.shape[1]),
    "testing_features": int(X_test_prepared.shape[1]),
    "removed_constant_features": int(len(removed_features)),
    "overlapping_feature_groups": int(len(overlap)),
    "missing_values_remaining": int(
        np.isnan(X_train_prepared).sum()
        + np.isnan(X_test_prepared).sum()
    ),
    "random_state": RANDOM_STATE,
    "split_method": "StratifiedGroupKFold",
    "imputation": "Median",
    "feature_selection": "VarianceThreshold"
}

with open(
    RESULTS_DIR / "phase10_summary.json",
    "w"
) as f:
    json.dump(summary, f, indent=4)

# =====================================================
# 12. FINAL REPORT
# =====================================================

print("\n" + "=" * 65)
print("PHASE 10 - FINAL SUMMARY")
print("=" * 65)

for key, value in summary.items():
    print(f"{key}: {value}")

print("\nFiles generated successfully:")
print("- data/prepared/prepared_dataset.joblib")
print("- models/preprocessing_pipeline.pkl")
print("- results/preparation/class_distribution_split.csv")
print("- results/preparation/selected_features.csv")
print("- results/preparation/removed_constant_features.csv")
print("- results/preparation/phase10_summary.json")

print("\nPHASE 10 DATA PREPARATION COMPLETED!")
print("=" * 65)
