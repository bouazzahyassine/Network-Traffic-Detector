import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve
)

# =========================================================
# PHASE 12 - MODEL EVALUATION AND OPTIMIZATION
# =========================================================

print("=" * 65)
print("PHASE 12 - MODEL EVALUATION AND OPTIMIZATION")
print("=" * 65)

os.makedirs("results/optimization", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ---------------------------------------------------------
# 1. LOAD PREPARED DATA
# ---------------------------------------------------------

print("\n[1] Loading prepared dataset...")

data = joblib.load("data/prepared/prepared_dataset.joblib")

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]
feature_names = list(data["feature_names"])

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# Binary labels for anomaly detection
y_train_binary = np.where(y_train == "BENIGN", 0, 1)
y_test_binary = np.where(y_test == "BENIGN", 0, 1)

# ---------------------------------------------------------
# 2. BASELINE RANDOM FOREST EVALUATION
# ---------------------------------------------------------

print("\n[2] Evaluating original Random Forest...")

original_rf = joblib.load("models/random_forest.pkl")

start = time.time()
baseline_predictions = original_rf.predict(X_test)
baseline_time = time.time() - start

baseline_accuracy = accuracy_score(y_test, baseline_predictions)
baseline_precision = precision_score(
    y_test, baseline_predictions, average="macro", zero_division=0
)
baseline_recall = recall_score(
    y_test, baseline_predictions, average="macro", zero_division=0
)
baseline_f1 = f1_score(
    y_test, baseline_predictions, average="macro", zero_division=0
)

print("Original Random Forest:")
print("Accuracy:", baseline_accuracy)
print("Precision Macro:", baseline_precision)
print("Recall Macro:", baseline_recall)
print("F1 Macro:", baseline_f1)

# ---------------------------------------------------------
# 3. RANDOM FOREST WITHOUT DESTINATION PORT
# ---------------------------------------------------------

print("\n[3] Testing Random Forest without Destination Port...")

if "Destination Port" not in feature_names:
    raise ValueError("Destination Port was not found in feature_names.")

port_index = feature_names.index("Destination Port")
selected_indices = [
    i for i in range(len(feature_names)) if i != port_index
]

X_train_no_port = X_train[:, selected_indices]
X_test_no_port = X_test[:, selected_indices]

print("Removed feature: Destination Port")
print("New number of features:", X_train_no_port.shape[1])

rf_no_port = RandomForestClassifier(
    n_estimators=100,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=2
)

start = time.time()
rf_no_port.fit(X_train_no_port, y_train)
rf_training_time = time.time() - start

rf_no_port_predictions = rf_no_port.predict(X_test_no_port)

rf_no_port_metrics = {
    "accuracy": float(accuracy_score(y_test, rf_no_port_predictions)),
    "precision_macro": float(precision_score(
        y_test, rf_no_port_predictions,
        average="macro", zero_division=0
    )),
    "recall_macro": float(recall_score(
        y_test, rf_no_port_predictions,
        average="macro", zero_division=0
    )),
    "f1_macro": float(f1_score(
        y_test, rf_no_port_predictions,
        average="macro", zero_division=0
    )),
    "training_time_seconds": round(rf_training_time, 2)
}

print("\nRandom Forest WITHOUT Destination Port:")
for key, value in rf_no_port_metrics.items():
    print(f"{key}: {value}")

print("\nClassification Report:")
print(classification_report(
    y_test, rf_no_port_predictions, zero_division=0
))

joblib.dump(
    rf_no_port,
    "models/random_forest_without_destination_port.pkl"
)

# Save RF confusion matrix
rf_cm = confusion_matrix(y_test, rf_no_port_predictions)
disp = ConfusionMatrixDisplay(
    confusion_matrix=rf_cm,
    display_labels=rf_no_port.classes_
)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Random Forest Without Destination Port")
plt.tight_layout()
plt.savefig(
    "results/optimization/rf_without_port_confusion_matrix.png",
    dpi=150
)
plt.close()

# Save RF classification report
rf_report = classification_report(
    y_test, rf_no_port_predictions,
    output_dict=True, zero_division=0
)
pd.DataFrame(rf_report).transpose().to_csv(
    "results/optimization/rf_without_port_report.csv"
)

# ---------------------------------------------------------
# 4. PREPARE VALIDATION SET FOR ISOLATION FOREST
# ---------------------------------------------------------

print("\n[4] Preparing validation set for Isolation Forest...")

X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train_binary,
    test_size=0.20,
    random_state=42,
    stratify=y_train_binary
)

# Isolation Forest is fitted only on BENIGN training samples
X_fit_benign = X_fit[y_fit == 0]

print("Internal training samples:", len(X_fit))
print("Validation samples:", len(X_val))
print("BENIGN samples used to fit Isolation Forest:", len(X_fit_benign))
print("ATTACK samples in validation:", int(np.sum(y_val == 1)))

# ---------------------------------------------------------
# 5. TRAIN ISOLATION FOREST
# ---------------------------------------------------------

print("\n[5] Training Isolation Forest...")

iso_model = IsolationForest(
    n_estimators=100,
    max_samples=256,
    contamination="auto",
    random_state=42,
    n_jobs=2
)

start = time.time()
iso_model.fit(X_fit_benign)
iso_training_time = time.time() - start

print("Isolation Forest training completed.")
print("Training time:", round(iso_training_time, 2), "seconds")

# ---------------------------------------------------------
# 6. OPTIMIZE THRESHOLD USING VALIDATION ONLY
# ---------------------------------------------------------

print("\n[6] Optimizing anomaly threshold using validation set...")

# Higher score = more anomalous
val_scores = -iso_model.decision_function(X_val)

precision_values, recall_values, thresholds = precision_recall_curve(
    y_val, val_scores
)

if len(thresholds) == 0:
    raise RuntimeError("No thresholds generated for validation.")

f1_values = (
    2 * precision_values[:-1] * recall_values[:-1]
    / (precision_values[:-1] + recall_values[:-1] + 1e-12)
)

best_index = int(np.argmax(f1_values))
best_threshold = float(thresholds[best_index])
best_validation_f1 = float(f1_values[best_index])

print("Selected threshold:", best_threshold)
print("Validation F1 Attack:", best_validation_f1)
print("Validation Precision Attack:", float(precision_values[best_index]))
print("Validation Recall Attack:", float(recall_values[best_index]))

# ---------------------------------------------------------
# 7. FINAL EVALUATION ON TEST SET
# ---------------------------------------------------------

print("\n[7] Evaluating optimized Isolation Forest on test set...")

test_scores = -iso_model.decision_function(X_test)

# Score >= threshold means ATTACK
iso_predictions = (test_scores >= best_threshold).astype(int)

iso_metrics = {
    "accuracy": float(accuracy_score(y_test_binary, iso_predictions)),
    "precision_attack": float(precision_score(
        y_test_binary, iso_predictions, zero_division=0
    )),
    "recall_attack": float(recall_score(
        y_test_binary, iso_predictions, zero_division=0
    )),
    "f1_attack": float(f1_score(
        y_test_binary, iso_predictions, zero_division=0
    )),
    "training_time_seconds": round(iso_training_time, 2),
    "optimized_threshold": best_threshold,
    "validation_f1_attack": best_validation_f1
}

print("\nOptimized Isolation Forest Metrics:")
for key, value in iso_metrics.items():
    print(f"{key}: {value}")

print("\nClassification Report:")
print(classification_report(
    y_test_binary,
    iso_predictions,
    labels=[0, 1],
    target_names=["BENIGN", "ATTACK"],
    zero_division=0
))

# ---------------------------------------------------------
# 8. SAVE MODELS AND REPORTS
# ---------------------------------------------------------

print("\n[8] Saving optimized models and reports...")

joblib.dump(
    iso_model,
    "models/isolation_forest_optimized.pkl"
)

# Save threshold separately
with open("models/isolation_forest_threshold.json", "w") as f:
    json.dump({"threshold": best_threshold}, f, indent=4)

iso_report = classification_report(
    y_test_binary,
    iso_predictions,
    labels=[0, 1],
    target_names=["BENIGN", "ATTACK"],
    output_dict=True,
    zero_division=0
)

pd.DataFrame(iso_report).transpose().to_csv(
    "results/optimization/isolation_forest_optimized_report.csv"
)

iso_cm = confusion_matrix(
    y_test_binary, iso_predictions, labels=[0, 1]
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=iso_cm,
    display_labels=["BENIGN", "ATTACK"]
)
disp.plot(cmap="Oranges")
plt.title("Optimized Isolation Forest")
plt.tight_layout()
plt.savefig(
    "results/optimization/isolation_forest_optimized_confusion_matrix.png",
    dpi=150
)
plt.close()

# ---------------------------------------------------------
# 9. SAVE COMPARISON
# ---------------------------------------------------------

comparison = {
    "Random Forest Original": {
        "accuracy": float(baseline_accuracy),
        "precision_macro": float(baseline_precision),
        "recall_macro": float(baseline_recall),
        "f1_macro": float(baseline_f1),
        "evaluation_time_seconds": round(baseline_time, 2)
    },
    "Random Forest Without Destination Port": rf_no_port_metrics,
    "Isolation Forest Optimized": iso_metrics
}

with open("results/optimization/phase12_comparison.json", "w") as f:
    json.dump(comparison, f, indent=4)

print("\n" + "=" * 65)
print("PHASE 12 - COMPLETED")
print("=" * 65)

print("\nGenerated files:")
print("- models/random_forest_without_destination_port.pkl")
print("- models/isolation_forest_optimized.pkl")
print("- models/isolation_forest_threshold.json")
print("- results/optimization/rf_without_port_report.csv")
print("- results/optimization/rf_without_port_confusion_matrix.png")
print("- results/optimization/isolation_forest_optimized_report.csv")
print("- results/optimization/isolation_forest_optimized_confusion_matrix.png")
print("- results/optimization/phase12_comparison.json")
