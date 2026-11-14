
#!/usr/bin/env python3

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix
)

# =========================================================
# PHASE 14.2 - GROUP-AWARE VALIDATION
# =========================================================

print("=" * 70)
print("PHASE 14.2 - GROUP-AWARE VALIDATION")
print("=" * 70)

DATA_PATH = "data/processed/Tuesday_clean.csv"
PREPARED_PATH = "data/prepared/prepared_dataset.joblib"
OUTPUT_DIR = "results/robustness"
MODEL_DIR = "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

print("\n[1] Loading datasets...")

df = pd.read_csv(DATA_PATH, low_memory=False)
prepared = joblib.load(PREPARED_PATH)

X_train = prepared["X_train"]
X_test = prepared["X_test"]

y_train = np.asarray(prepared["y_train"])
y_test = np.asarray(prepared["y_test"])

train_indices = np.asarray(prepared["train_indices"])
test_indices = np.asarray(prepared["test_indices"])

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# Binary labels
y_train_binary = (y_train != "BENIGN").astype(int)
y_test_binary = (y_test != "BENIGN").astype(int)

# ---------------------------------------------------------
# 2. RECREATE DUPLICATE GROUPS
# ---------------------------------------------------------

print("\n[2] Recreating duplicate groups...")

if "Label" not in df.columns:
    raise ValueError("Label column missing.")

X_original = df.drop(columns=["Label"]).copy()

for col in X_original.columns:
    X_original[col] = pd.to_numeric(
        X_original[col],
        errors="coerce"
    )

X_original = X_original.replace(
    [np.inf, -np.inf],
    np.nan
)

all_groups = pd.util.hash_pandas_object(
    X_original,
    index=False
).to_numpy(dtype="uint64")

groups_train = all_groups[train_indices]

if len(groups_train) != len(X_train):
    raise ValueError("Group and training dimensions mismatch.")

print("Unique training groups:", len(np.unique(groups_train)))

# ---------------------------------------------------------
# 3. GROUP-AWARE SPLIT
# ---------------------------------------------------------

print("\n[3] Creating group-aware validation split...")

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

fit_positions, val_positions = next(
    cv.split(
        X_train,
        y_train_binary,
        groups_train
    )
)

X_fit = X_train[fit_positions]
X_val = X_train[val_positions]

y_fit = y_train_binary[fit_positions]
y_val = y_train_binary[val_positions]

groups_fit = groups_train[fit_positions]
groups_val = groups_train[val_positions]

print("Fit rows:", len(X_fit))
print("Validation rows:", len(X_val))

print("Fit BENIGN:", int(np.sum(y_fit == 0)))
print("Fit ATTACK:", int(np.sum(y_fit == 1)))

print("Validation BENIGN:", int(np.sum(y_val == 0)))
print("Validation ATTACK:", int(np.sum(y_val == 1)))

# ---------------------------------------------------------
# 4. VERIFY GROUP SEPARATION
# ---------------------------------------------------------

print("\n[4] Verifying duplicate group separation...")

overlap = np.intersect1d(
    np.unique(groups_fit),
    np.unique(groups_val)
)

print("Unique fit groups:", len(np.unique(groups_fit)))
print("Unique validation groups:", len(np.unique(groups_val)))
print("Overlapping groups:", len(overlap))

if len(overlap) != 0:
    raise RuntimeError(
        "Duplicate leakage detected! Stopping execution."
    )

print("Group separation: PASSED")

# Save overlap audit
overlap_df = pd.DataFrame({
    "overlapping_group_id": overlap
})

overlap_df.to_csv(
    f"{OUTPUT_DIR}/phase14_2_group_overlap.csv",
    index=False
)

# ---------------------------------------------------------
# 5. TRAIN ONLY ON BENIGN SAMPLES
# ---------------------------------------------------------

print("\n[5] Training Isolation Forest...")

X_fit_benign = X_fit[y_fit == 0]

if len(X_fit_benign) == 0:
    raise ValueError("No BENIGN samples available for training.")

iso_model = IsolationForest(
    n_estimators=100,
    max_samples=256,
    contamination="auto",
    random_state=42,
    n_jobs=2
)

start = time.time()

iso_model.fit(X_fit_benign)

training_time = time.time() - start

print("Training completed.")
print("BENIGN training samples:", len(X_fit_benign))
print("Training time:", round(training_time, 2), "seconds")

# ---------------------------------------------------------
# 6. OPTIMIZE THRESHOLD ON VALIDATION
# ---------------------------------------------------------

print("\n[6] Optimizing threshold on validation...")

val_scores = -iso_model.decision_function(X_val)

precision_values, recall_values, thresholds = (
    precision_recall_curve(y_val, val_scores)
)

if len(thresholds) == 0:
    raise RuntimeError("No thresholds generated.")

f1_values = (
    2 * precision_values[:-1] * recall_values[:-1]
    / (
        precision_values[:-1]
        + recall_values[:-1]
        + 1e-12
    )
)

best_index = int(np.argmax(f1_values))

best_threshold = float(thresholds[best_index])
best_validation_f1 = float(f1_values[best_index])

best_validation_precision = float(
    precision_values[best_index]
)

best_validation_recall = float(
    recall_values[best_index]
)

print("Best threshold:", best_threshold)
print("Validation Precision:", best_validation_precision)
print("Validation Recall:", best_validation_recall)
print("Validation F1:", best_validation_f1)

# ---------------------------------------------------------
# 7. FINAL TEST EVALUATION
# ---------------------------------------------------------

print("\n[7] Evaluating on untouched test set...")

test_scores = -iso_model.decision_function(X_test)

test_predictions = (
    test_scores >= best_threshold
).astype(int)

test_metrics = {
    "accuracy": float(
        accuracy_score(y_test_binary, test_predictions)
    ),
    "precision_attack": float(
        precision_score(
            y_test_binary,
            test_predictions,
            zero_division=0
        )
    ),
    "recall_attack": float(
        recall_score(
            y_test_binary,
            test_predictions,
            zero_division=0
        )
    ),
    "f1_attack": float(
        f1_score(
            y_test_binary,
            test_predictions,
            zero_division=0
        )
    )
}

cm = confusion_matrix(
    y_test_binary,
    test_predictions,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

print("\nTEST METRICS")

for key, value in test_metrics.items():
    print(f"{key}: {value:.6f}")

print("\nConfusion Matrix:")
print(cm)

print("\nTN:", int(tn))
print("FP:", int(fp))
print("FN:", int(fn))
print("TP:", int(tp))

print("\nClassification Report:")

report = classification_report(
    y_test_binary,
    test_predictions,
    labels=[0, 1],
    target_names=["BENIGN", "ATTACK"],
    output_dict=True,
    zero_division=0
)

print(
    classification_report(
        y_test_binary,
        test_predictions,
        labels=[0, 1],
        target_names=["BENIGN", "ATTACK"],
        zero_division=0
    )
)

# ---------------------------------------------------------
# 8. SAVE RESULTS
# ---------------------------------------------------------

print("\n[8] Saving results...")

model_path = (
    f"{MODEL_DIR}/isolation_forest_grouped.pkl"
)

threshold_path = (
    f"{MODEL_DIR}/isolation_forest_grouped_threshold.json"
)

joblib.dump(iso_model, model_path)

threshold_data = {
    "threshold": best_threshold,
    "validation_f1": best_validation_f1,
    "validation_precision": best_validation_precision,
    "validation_recall": best_validation_recall,
    "method": "StratifiedGroupKFold",
    "random_state": 42
}

with open(threshold_path, "w") as f:
    json.dump(threshold_data, f, indent=4)

pd.DataFrame(report).transpose().to_csv(
    f"{OUTPUT_DIR}/phase14_2_classification_report.csv"
)

summary = {
    "phase": "14.2",
    "title": "Group-Aware Validation for Isolation Forest",

    "split_method": "StratifiedGroupKFold",
    "n_splits": 5,
    "random_state": 42,

    "fit_rows": int(len(X_fit)),
    "validation_rows": int(len(X_val)),
    "fit_benign_rows": int(len(X_fit_benign)),

    "fit_unique_groups": int(len(np.unique(groups_fit))),
    "validation_unique_groups": int(len(np.unique(groups_val))),
    "overlapping_groups": int(len(overlap)),

    "threshold": best_threshold,
    "validation_f1": best_validation_f1,
    "validation_precision": best_validation_precision,
    "validation_recall": best_validation_recall,

    "training_time_seconds": round(training_time, 4),

    "test_metrics": test_metrics,

    "confusion_matrix": {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    },

    "integrity": {
        "group_separation_passed": bool(len(overlap) == 0),
        "test_used_for_threshold_selection": False
    }
}

summary_path = (
    f"{OUTPUT_DIR}/phase14_2_grouped_validation_summary.json"
)

with open(summary_path, "w") as f:
    json.dump(summary, f, indent=4)

# ---------------------------------------------------------
# 9. CONFUSION MATRIX VISUALIZATION
# ---------------------------------------------------------

plt.figure(figsize=(7, 6))

plt.imshow(cm, interpolation="nearest")
plt.title("Phase 14.2 - Isolation Forest")
plt.colorbar()

plt.xticks([0, 1], ["BENIGN", "ATTACK"])
plt.yticks([0, 1], ["BENIGN", "ATTACK"])

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/phase14_2_confusion_matrix.png",
    dpi=300
)

plt.close()

# ---------------------------------------------------------
# 10. COMPLETION
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("PHASE 14.2 COMPLETED")
print("=" * 70)

print("\nGroup overlap:", len(overlap))
print("Optimized threshold:", best_threshold)
print("Test F1 Attack:", test_metrics["f1_attack"])

print("\nSaved files:")
print(model_path)
print(threshold_path)
print(summary_path)
print(f"{OUTPUT_DIR}/phase14_2_group_overlap.csv")
print(f"{OUTPUT_DIR}/phase14_2_classification_report.csv")
print(f"{OUTPUT_DIR}/phase14_2_confusion_matrix.png")
