
# ============================================================
# PHASE 13 - ERROR ANALYSIS AND MODEL INTERPRETABILITY
# Network Traffic Anomaly Detection
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score
)

# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

DATA_PATH = "data/prepared/prepared_dataset.joblib"

RF_PATH = "models/random_forest.pkl"
ISO_PATH = "models/isolation_forest_optimized.pkl"
THRESHOLD_PATH = "models/isolation_forest_threshold.json"

OUTPUT_DIR = "results/error_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("PHASE 13 - ERROR ANALYSIS AND MODEL INTERPRETABILITY")
print("=" * 65)

# ------------------------------------------------------------
# 2. LOAD PREPARED DATASET
# ------------------------------------------------------------

print("\n[1] Loading prepared dataset...")

data = joblib.load(DATA_PATH)

X_test = data["X_test"]
y_test = np.asarray(data["y_test"])
feature_names = list(data["feature_names"])
test_indices = np.asarray(data["test_indices"])

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
print("Features:", len(feature_names))

# ------------------------------------------------------------
# 3. LOAD TRAINED MODELS
# ------------------------------------------------------------

print("\n[2] Loading trained models...")

rf_model = joblib.load(RF_PATH)
iso_model = joblib.load(ISO_PATH)

with open(THRESHOLD_PATH, "r") as f:
    threshold = float(json.load(f)["threshold"])

print("Random Forest loaded.")
print("Isolation Forest loaded.")
print("Optimized threshold:", threshold)

# ============================================================
# PART A - RANDOM FOREST ERROR ANALYSIS
# ============================================================

print("\n[3] Random Forest error analysis...")

# Predictions
rf_predictions = rf_model.predict(X_test)

# Metrics
rf_accuracy = accuracy_score(y_test, rf_predictions)

rf_precision = precision_score(
    y_test,
    rf_predictions,
    average="macro",
    zero_division=0
)

rf_recall = recall_score(
    y_test,
    rf_predictions,
    average="macro",
    zero_division=0
)

rf_f1 = f1_score(
    y_test,
    rf_predictions,
    average="macro",
    zero_division=0
)

print("\nRandom Forest Metrics:")
print("Accuracy:", rf_accuracy)
print("Precision Macro:", rf_precision)
print("Recall Macro:", rf_recall)
print("F1 Macro:", rf_f1)

# ------------------------------------------------------------
# 4. RANDOM FOREST CLASSIFICATION REPORT
# ------------------------------------------------------------

rf_report = classification_report(
    y_test,
    rf_predictions,
    output_dict=True,
    zero_division=0
)

pd.DataFrame(rf_report).transpose().to_csv(
    f"{OUTPUT_DIR}/random_forest_classification_report.csv"
)

# ------------------------------------------------------------
# 5. RANDOM FOREST CONFUSION MATRIX
# ------------------------------------------------------------

print("\n[4] Generating Random Forest confusion matrix...")

classes = rf_model.classes_

cm_rf = confusion_matrix(
    y_test,
    rf_predictions,
    labels=classes
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_rf,
    display_labels=classes
)

disp.plot(cmap="Blues", xticks_rotation=45)

plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/random_forest_confusion_matrix.png",
    dpi=150
)

plt.close()

# ------------------------------------------------------------
# 6. RANDOM FOREST ERROR EXTRACTION
# ------------------------------------------------------------

print("\n[5] Extracting Random Forest errors...")

rf_error_mask = y_test != rf_predictions

rf_errors = pd.DataFrame({
    "test_index": test_indices[rf_error_mask],
    "true_label": y_test[rf_error_mask],
    "predicted_label": rf_predictions[rf_error_mask]
})

rf_errors.to_csv(
    f"{OUTPUT_DIR}/random_forest_errors.csv",
    index=False
)

# One-vs-rest false positives and false negatives
rf_fp_records = []
rf_fn_records = []

for class_name in classes:

    fp_mask = (
        (y_test != class_name) &
        (rf_predictions == class_name)
    )

    fn_mask = (
        (y_test == class_name) &
        (rf_predictions != class_name)
    )

    for idx in np.where(fp_mask)[0]:
        rf_fp_records.append({
            "test_index": test_indices[idx],
            "true_label": y_test[idx],
            "predicted_label": rf_predictions[idx],
            "analyzed_class": class_name
        })

    for idx in np.where(fn_mask)[0]:
        rf_fn_records.append({
            "test_index": test_indices[idx],
            "true_label": y_test[idx],
            "predicted_label": rf_predictions[idx],
            "analyzed_class": class_name
        })

pd.DataFrame(rf_fp_records).to_csv(
    f"{OUTPUT_DIR}/random_forest_false_positives.csv",
    index=False
)

pd.DataFrame(rf_fn_records).to_csv(
    f"{OUTPUT_DIR}/random_forest_false_negatives.csv",
    index=False
)

print("Total misclassified samples:", int(rf_error_mask.sum()))
print("One-vs-rest FP records:", len(rf_fp_records))
print("One-vs-rest FN records:", len(rf_fn_records))

# ------------------------------------------------------------
# 7. RANDOM FOREST FEATURE IMPORTANCE
# ------------------------------------------------------------

print("\n[6] Analyzing feature importance...")

feature_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

feature_importance.to_csv(
    f"{OUTPUT_DIR}/random_forest_feature_importance.csv",
    index=False
)

print("\nTop 10 features:")
print(feature_importance.head(10).to_string(index=False))

# Plot top 20 features
top_features = feature_importance.head(20).sort_values(
    by="importance",
    ascending=True
)

plt.figure(figsize=(10, 8))

plt.barh(
    top_features["feature"],
    top_features["importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Random Forest - Top 20 Important Features")
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/random_forest_top_features.png",
    dpi=150
)

plt.close()

# ------------------------------------------------------------
# 8. RANDOM FOREST PRECISION-RECALL CURVES
# ------------------------------------------------------------

print("\n[7] Generating Random Forest Precision-Recall curves...")

rf_probabilities = rf_model.predict_proba(X_test)

plt.figure(figsize=(9, 7))

rf_ap_by_class = {}

for i, class_name in enumerate(classes):

    y_binary = (y_test == class_name).astype(int)

    precision_curve, recall_curve, _ = precision_recall_curve(
        y_binary,
        rf_probabilities[:, i]
    )

    ap = average_precision_score(
        y_binary,
        rf_probabilities[:, i]
    )

    rf_ap_by_class[str(class_name)] = float(ap)

    plt.plot(
        recall_curve,
        precision_curve,
        label=f"{class_name} (AP={ap:.4f})"
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Random Forest - Precision-Recall Curves")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/random_forest_precision_recall.png",
    dpi=150
)

plt.close()

# ============================================================
# PART B - ISOLATION FOREST ERROR ANALYSIS
# ============================================================

print("\n[8] Isolation Forest error analysis...")

# IMPORTANT:
# Phase 12 used the negative decision_function.
# Higher anomaly score = more anomalous.
# Keep exactly the same calculation here.

iso_scores = -iso_model.decision_function(X_test)

# Same threshold and comparison as Phase 12
iso_predictions = np.where(
    iso_scores >= threshold,
    "ATTACK",
    "BENIGN"
)

# Convert multiclass labels to binary labels
iso_true = np.where(
    y_test == "BENIGN",
    "BENIGN",
    "ATTACK"
)

# ------------------------------------------------------------
# 9. ISOLATION FOREST METRICS
# ------------------------------------------------------------

iso_accuracy = accuracy_score(
    iso_true,
    iso_predictions
)

iso_precision = precision_score(
    iso_true,
    iso_predictions,
    pos_label="ATTACK",
    zero_division=0
)

iso_recall = recall_score(
    iso_true,
    iso_predictions,
    pos_label="ATTACK",
    zero_division=0
)

iso_f1 = f1_score(
    iso_true,
    iso_predictions,
    pos_label="ATTACK",
    zero_division=0
)

print("\nIsolation Forest Metrics:")
print("Accuracy:", iso_accuracy)
print("Precision Attack:", iso_precision)
print("Recall Attack:", iso_recall)
print("F1 Attack:", iso_f1)

# ------------------------------------------------------------
# 10. ISOLATION FOREST CLASSIFICATION REPORT
# ------------------------------------------------------------

iso_report = classification_report(
    iso_true,
    iso_predictions,
    labels=["BENIGN", "ATTACK"],
    output_dict=True,
    zero_division=0
)

pd.DataFrame(iso_report).transpose().to_csv(
    f"{OUTPUT_DIR}/isolation_forest_classification_report.csv"
)

# ------------------------------------------------------------
# 11. ISOLATION FOREST CONFUSION MATRIX
# ------------------------------------------------------------

print("\n[9] Generating Isolation Forest confusion matrix...")

iso_classes = ["BENIGN", "ATTACK"]

cm_iso = confusion_matrix(
    iso_true,
    iso_predictions,
    labels=iso_classes
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_iso,
    display_labels=iso_classes
)

disp.plot(cmap="Oranges")

plt.title("Isolation Forest - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/isolation_forest_confusion_matrix.png",
    dpi=150
)

plt.close()

# ------------------------------------------------------------
# 12. ISOLATION FOREST FALSE POSITIVES AND NEGATIVES
# ------------------------------------------------------------

print("\n[10] Extracting Isolation Forest errors...")

iso_fp_mask = (
    (iso_true == "BENIGN") &
    (iso_predictions == "ATTACK")
)

iso_fn_mask = (
    (iso_true == "ATTACK") &
    (iso_predictions == "BENIGN")
)

iso_error_mask = iso_fp_mask | iso_fn_mask

iso_errors = pd.DataFrame({
    "test_index": test_indices[iso_error_mask],
    "true_label": iso_true[iso_error_mask],
    "predicted_label": iso_predictions[iso_error_mask],
    "anomaly_score": iso_scores[iso_error_mask],
    "error_type": np.where(
        iso_fp_mask[iso_error_mask],
        "FALSE_POSITIVE",
        "FALSE_NEGATIVE"
    )
})

iso_errors.to_csv(
    f"{OUTPUT_DIR}/isolation_forest_errors.csv",
    index=False
)

iso_fp_df = pd.DataFrame({
    "test_index": test_indices[iso_fp_mask],
    "true_label": iso_true[iso_fp_mask],
    "predicted_label": iso_predictions[iso_fp_mask],
    "original_attack_label": y_test[iso_fp_mask],
    "anomaly_score": iso_scores[iso_fp_mask]
})

iso_fp_df.to_csv(
    f"{OUTPUT_DIR}/isolation_forest_false_positives.csv",
    index=False
)

iso_fn_df = pd.DataFrame({
    "test_index": test_indices[iso_fn_mask],
    "true_label": iso_true[iso_fn_mask],
    "predicted_label": iso_predictions[iso_fn_mask],
    "original_attack_label": y_test[iso_fn_mask],
    "anomaly_score": iso_scores[iso_fn_mask]
})

iso_fn_df.to_csv(
    f"{OUTPUT_DIR}/isolation_forest_false_negatives.csv",
    index=False
)

print("False Positives:", int(iso_fp_mask.sum()))
print("False Negatives:", int(iso_fn_mask.sum()))

# ------------------------------------------------------------
# 13. ISOLATION FOREST PRECISION-RECALL CURVE
# ------------------------------------------------------------

print("\n[11] Generating Isolation Forest Precision-Recall curve...")

iso_y_binary = (iso_true == "ATTACK").astype(int)

# iso_scores is already the negative decision_function.
# Higher score means more anomalous.

iso_precision_curve, iso_recall_curve, _ = precision_recall_curve(
    iso_y_binary,
    iso_scores
)

iso_ap = average_precision_score(
    iso_y_binary,
    iso_scores
)

plt.figure(figsize=(9, 7))

plt.plot(
    iso_recall_curve,
    iso_precision_curve,
    label=f"Isolation Forest (AP={iso_ap:.4f})"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Isolation Forest - Precision-Recall Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/isolation_forest_precision_recall.png",
    dpi=150
)

plt.close()

# ============================================================
# PART C - FINAL SUMMARY
# ============================================================

print("\n[12] Generating final summary...")

summary = {
    "phase": 13,
    "title": "Error Analysis and Model Interpretability",

    "dataset": {
        "test_samples": int(len(y_test)),
        "features": int(len(feature_names))
    },

    "random_forest": {
        "accuracy": float(rf_accuracy),
        "precision_macro": float(rf_precision),
        "recall_macro": float(rf_recall),
        "f1_macro": float(rf_f1),
        "misclassified_samples": int(rf_error_mask.sum()),
        "false_positive_records": int(len(rf_fp_records)),
        "false_negative_records": int(len(rf_fn_records)),
        "average_precision_by_class": rf_ap_by_class,
        "top_10_features": feature_importance.head(10).to_dict(
            orient="records"
        )
    },

    "isolation_forest": {
        "threshold": float(threshold),
        "accuracy": float(iso_accuracy),
        "precision_attack": float(iso_precision),
        "recall_attack": float(iso_recall),
        "f1_attack": float(iso_f1),
        "false_positives": int(iso_fp_mask.sum()),
        "false_negatives": int(iso_fn_mask.sum()),
        "average_precision": float(iso_ap)
    }
}

with open(
    f"{OUTPUT_DIR}/phase13_summary.json",
    "w"
) as f:
    json.dump(summary, f, indent=4)

# ------------------------------------------------------------
# 14. FINAL RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("PHASE 13 - COMPLETED")
print("=" * 65)

print("\nRandom Forest:")
print("Accuracy:", round(rf_accuracy, 6))
print("F1 Macro:", round(rf_f1, 6))
print("Misclassified:", int(rf_error_mask.sum()))

print("\nIsolation Forest:")
print("Accuracy:", round(iso_accuracy, 6))
print("Precision Attack:", round(iso_precision, 6))
print("Recall Attack:", round(iso_recall, 6))
print("F1 Attack:", round(iso_f1, 6))
print("False Positives:", int(iso_fp_mask.sum()))
print("False Negatives:", int(iso_fn_mask.sum()))

print("\nGenerated files:")

for filename in sorted(os.listdir(OUTPUT_DIR)):
    print("-", filename)

print("\nPhase 13 analysis completed successfully.")
