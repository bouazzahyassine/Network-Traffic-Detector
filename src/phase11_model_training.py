
#!/usr/bin/env python3

"""
PHASE 11 - MACHINE LEARNING MODEL TRAINING

Models:
1. Random Forest - Multiclass Classification
2. Isolation Forest - Binary Anomaly Detection

Dataset: CIC-IDS2017 Tuesday
"""

from pathlib import Path
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =====================================================
# 1. PATHS
# =====================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data/prepared/prepared_dataset.joblib"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results/training"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("PHASE 11 - MACHINE LEARNING MODEL TRAINING")
print("=" * 65)

# =====================================================
# 2. LOAD PREPARED DATA
# =====================================================

print("\n[1] Loading prepared dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(DATA_PATH)

data = joblib.load(DATA_PATH)

X_train = data["X_train"]
X_test = data["X_test"]

y_train = np.asarray(data["y_train"])
y_test = np.asarray(data["y_test"])

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("Training classes:", np.unique(y_train))

# =====================================================
# 3. RANDOM FOREST
# =====================================================

print("\n[2] Training Random Forest...")

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=2
)

start = time.time()

rf.fit(X_train, y_train)

rf_training_time = time.time() - start

print("Random Forest training completed!")
print("Training time:", round(rf_training_time, 2), "seconds")

# =====================================================
# 4. RANDOM FOREST EVALUATION
# =====================================================

print("\n[3] Evaluating Random Forest...")

rf_predictions = rf.predict(X_test)

rf_metrics = {
    "accuracy": float(accuracy_score(y_test, rf_predictions)),
    "precision_macro": float(
        precision_score(y_test, rf_predictions, average="macro", zero_division=0)
    ),
    "recall_macro": float(
        recall_score(y_test, rf_predictions, average="macro", zero_division=0)
    ),
    "f1_macro": float(
        f1_score(y_test, rf_predictions, average="macro", zero_division=0)
    ),
    "training_time_seconds": round(rf_training_time, 2)
}

print("\nRandom Forest Metrics:")
for name, value in rf_metrics.items():
    print(f"{name}: {value}")

print("\nClassification Report:")
print(classification_report(y_test, rf_predictions, zero_division=0))

rf_report = classification_report(
    y_test,
    rf_predictions,
    output_dict=True,
    zero_division=0
)

pd.DataFrame(rf_report).transpose().to_csv(
    RESULTS_DIR / "random_forest_classification_report.csv"
)

# Confusion matrix
rf_cm = confusion_matrix(y_test, rf_predictions)

fig, ax = plt.subplots(figsize=(8, 6))

ConfusionMatrixDisplay(
    confusion_matrix=rf_cm,
    display_labels=rf.classes_
).plot(ax=ax, cmap="Blues", values_format="d")

plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "random_forest_confusion_matrix.png", dpi=150)
plt.close()

# Feature importance
importance = pd.DataFrame({
    "Feature": data["feature_names"],
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)

importance.to_csv(
    RESULTS_DIR / "random_forest_feature_importance.csv",
    index=False
)

# Save model
joblib.dump(rf, MODELS_DIR / "random_forest.pkl")

print("Random Forest model saved!")

# =====================================================
# 5. ISOLATION FOREST
# =====================================================

print("\n[4] Preparing Isolation Forest...")

# Train only on BENIGN traffic
benign_mask = y_train == "BENIGN"

X_train_benign = X_train[benign_mask]

print("BENIGN training samples:", len(X_train_benign))

if len(X_train_benign) == 0:
    raise ValueError("No BENIGN training samples found.")

print("\n[5] Training Isolation Forest...")

iso = IsolationForest(
    n_estimators=100,
    max_samples=256,
    contamination="auto",
    random_state=42,
    n_jobs=2
)

start = time.time()

iso.fit(X_train_benign)

iso_training_time = time.time() - start

print("Isolation Forest training completed!")
print("Training time:", round(iso_training_time, 2), "seconds")

# =====================================================
# 6. ISOLATION FOREST EVALUATION
# =====================================================

print("\n[6] Evaluating Isolation Forest...")

# Isolation Forest:
#  1  = Normal
# -1  = Anomaly

iso_raw_predictions = iso.predict(X_test)

iso_predictions = np.where(
    iso_raw_predictions == 1,
    "BENIGN",
    "ATTACK"
)

# Convert real labels to binary classes
y_test_binary = np.where(
    y_test == "BENIGN",
    "BENIGN",
    "ATTACK"
)

iso_metrics = {
    "accuracy": float(
        accuracy_score(y_test_binary, iso_predictions)
    ),
    "precision_attack": float(
        precision_score(
            y_test_binary,
            iso_predictions,
            pos_label="ATTACK",
            zero_division=0
        )
    ),
    "recall_attack": float(
        recall_score(
            y_test_binary,
            iso_predictions,
            pos_label="ATTACK",
            zero_division=0
        )
    ),
    "f1_attack": float(
        f1_score(
            y_test_binary,
            iso_predictions,
            pos_label="ATTACK",
            zero_division=0
        )
    ),
    "training_time_seconds": round(iso_training_time, 2)
}

print("\nIsolation Forest Metrics:")

for name, value in iso_metrics.items():
    print(f"{name}: {value}")

print("\nClassification Report:")

print(classification_report(
    y_test_binary,
    iso_predictions,
    labels=["BENIGN", "ATTACK"],
    zero_division=0
))

iso_report = classification_report(
    y_test_binary,
    iso_predictions,
    labels=["BENIGN", "ATTACK"],
    output_dict=True,
    zero_division=0
)

pd.DataFrame(iso_report).transpose().to_csv(
    RESULTS_DIR / "isolation_forest_classification_report.csv"
)

# Confusion matrix
iso_cm = confusion_matrix(
    y_test_binary,
    iso_predictions,
    labels=["BENIGN", "ATTACK"]
)

fig, ax = plt.subplots(figsize=(7, 6))

ConfusionMatrixDisplay(
    confusion_matrix=iso_cm,
    display_labels=["BENIGN", "ATTACK"]
).plot(ax=ax, cmap="Oranges", values_format="d")

plt.title("Isolation Forest - Confusion Matrix")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "isolation_forest_confusion_matrix.png", dpi=150)
plt.close()

# Save model
joblib.dump(iso, MODELS_DIR / "isolation_forest.pkl")

print("Isolation Forest model saved!")

# =====================================================
# 7. SAVE COMPARISON
# =====================================================

print("\n[7] Saving model comparison...")

comparison = {
    "Random Forest": rf_metrics,
    "Isolation Forest": iso_metrics
}

with open(
    RESULTS_DIR / "phase11_metrics.json",
    "w"
) as f:
    json.dump(comparison, f, indent=4)

print("\n" + "=" * 65)
print("PHASE 11 - TRAINING COMPLETED")
print("=" * 65)

print("\nGenerated files:")
print("- models/random_forest.pkl")
print("- models/isolation_forest.pkl")
print("- results/training/random_forest_classification_report.csv")
print("- results/training/isolation_forest_classification_report.csv")
print("- results/training/random_forest_confusion_matrix.png")
print("- results/training/isolation_forest_confusion_matrix.png")
print("- results/training/random_forest_feature_importance.csv")
print("- results/training/phase11_metrics.json")

print("\nPHASE 11 COMPLETED SUCCESSFULLY!")
