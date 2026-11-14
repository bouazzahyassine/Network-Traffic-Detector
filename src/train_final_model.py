import os
import json
import time
import gc
import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

print("=" * 70)
print("FINAL RANDOM FOREST MODEL TRAINING")
print("=" * 70)

DATA_FILE = "data/processed/Tuesday_clean.csv"
PREPARED_FILE = "data/prepared/prepared_dataset.joblib"
MODEL_FILE = "models/random_forest_final_pipeline.pkl"
METADATA_FILE = "models/random_forest_final_metadata.json"

os.makedirs("models", exist_ok=True)

# 1. Load the original cleaned dataset and training indices
print("\n[1] Loading training data...")

df = pd.read_csv(DATA_FILE)
prepared = joblib.load(PREPARED_FILE)

train_indices = np.asarray(prepared["train_indices"])
y_train = np.asarray(prepared["y_train"]).astype(str)

train_df = df.iloc[train_indices].copy()

if len(train_df) != len(y_train):
    raise ValueError("Training data and labels do not match.")

if not np.array_equal(
    train_df["Label"].astype(str).to_numpy(),
    y_train
):
    raise ValueError("Label mismatch detected.")

X_train = train_df.drop(columns=["Label"])

del df, train_df, prepared
gc.collect()

# 2. Clean numerical features
print("\n[2] Preparing original features...")

for col in X_train.columns:
    X_train[col] = pd.to_numeric(
        X_train[col], errors="coerce"
    )

X_train = X_train.replace(
    [np.inf, -np.inf], np.nan
)

X_train = X_train.astype(np.float32)

original_features = list(X_train.columns)

print("Training samples:", len(X_train))
print("Original features:", len(original_features))
print("Classes:", np.unique(y_train))

# 3. Build the final pipeline
print("\n[3] Building final pipeline...")

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("variance", VarianceThreshold(threshold=0.0)),
    ("model", RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=1
    ))
])

# 4. Train on the complete training split
print("\n[4] Training Random Forest...")
start = time.time()

pipeline.fit(X_train, y_train)

elapsed = time.time() - start

print(f"Training completed in {elapsed:.2f} seconds.")

# 5. Verify the trained pipeline
print("\n[5] Verifying final model...")

selector = pipeline.named_steps["variance"]
model = pipeline.named_steps["model"]

selected_features = [
    name for name, selected
    in zip(original_features, selector.get_support())
    if selected
]

print("Original features:", len(original_features))
print("Selected features:", len(selected_features))
print("Model input features:", model.n_features_in_)
print("Model classes:", model.classes_)
print("Number of trees:", model.n_estimators)

if len(selected_features) != model.n_features_in_:
    raise ValueError("Feature compatibility error.")

# 6. Save final pipeline
print("\n[6] Saving final model...")

joblib.dump(pipeline, MODEL_FILE, compress=3)

metadata = {
    "project": "Network Traffic Anomaly Detection",
    "model": "RandomForestClassifier",
    "dataset": "CIC-IDS2017 Tuesday Working Hours",
    "training_samples": int(len(y_train)),
    "original_features": original_features,
    "selected_features": selected_features,
    "number_of_original_features": len(original_features),
    "number_of_selected_features": len(selected_features),
    "classes": model.classes_.tolist(),
    "parameters": {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_leaf": 2,
        "class_weight": "balanced_subsample",
        "random_state": 42,
        "n_jobs": 1
    },
    "training_time_seconds": round(elapsed, 2),
    "cross_validation_reference": "Phase 14.4"
}

with open(METADATA_FILE, "w") as f:
    json.dump(metadata, f, indent=4)

print("\n========== FINAL RESULT ==========")
print("Model:", MODEL_FILE)
print("Metadata:", METADATA_FILE)
print("Training samples:", len(y_train))
print("Selected features:", len(selected_features))
print("Training time:", round(elapsed, 2), "seconds")
print("Status: FINAL MODEL TRAINED AND SAVED")
