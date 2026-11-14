import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

DATA_FILE = "data/processed/Tuesday_clean.csv"
PREPARED_FILE = "data/prepared/prepared_dataset.joblib"
MODEL_FILE = "models/random_forest_final_pipeline.pkl"
OUTPUT_DIR = "results/final_evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("FINAL RANDOM FOREST - INDEPENDENT TEST EVALUATION")
print("=" * 70)

# 1. Load data and final model
df = pd.read_csv(DATA_FILE)
prepared = joblib.load(PREPARED_FILE)
pipeline = joblib.load(MODEL_FILE)

test_indices = np.asarray(prepared["test_indices"])
y_test = np.asarray(prepared["y_test"]).astype(str)

test_df = df.iloc[test_indices].copy()

if not np.array_equal(
    test_df["Label"].astype(str).to_numpy(), y_test
):
    raise ValueError("Test labels do not match.")

X_test = test_df.drop(columns=["Label"])

for col in X_test.columns:
    X_test[col] = pd.to_numeric(
        X_test[col], errors="coerce"
    )

X_test = X_test.replace(
    [np.inf, -np.inf], np.nan
)

X_test = X_test.astype(np.float32)

print("\n[1] Test data loaded")
print("Test samples:", len(y_test))
print("Original features:", X_test.shape[1])

# 2. Predict - no fitting or parameter adjustment
print("\n[2] Predicting test observations...")
y_pred = pipeline.predict(X_test)

# 3. Calculate metrics
labels = ["BENIGN", "FTP-Patator", "SSH-Patator"]

accuracy = accuracy_score(y_test, y_pred)
precision_macro = precision_score(
    y_test, y_pred, average="macro", zero_division=0
)
recall_macro = recall_score(
    y_test, y_pred, average="macro", zero_division=0
)
f1_macro = f1_score(
    y_test, y_pred, average="macro", zero_division=0
)
f1_weighted = f1_score(
    y_test, y_pred, average="weighted", zero_division=0
)

cm = confusion_matrix(y_test, y_pred, labels=labels)

report = classification_report(
    y_test,
    y_pred,
    labels=labels,
    target_names=labels,
    output_dict=True,
    zero_division=0
)

# 4. Display results
print("\n========== FINAL TEST METRICS ==========")
print(f"Accuracy       : {accuracy * 100:.6f}%")
print(f"Precision macro: {precision_macro * 100:.6f}%")
print(f"Recall macro   : {recall_macro * 100:.6f}%")
print(f"F1 macro       : {f1_macro * 100:.6f}%")
print(f"F1 weighted    : {f1_weighted * 100:.6f}%")

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(
    y_test, y_pred,
    labels=labels,
    target_names=labels,
    zero_division=0
))

print("\n========== CONFUSION MATRIX ==========")
print("Rows = actual classes; columns = predicted classes")
print(pd.DataFrame(cm, index=labels, columns=labels))

# 5. Save results
metrics = {
    "evaluation": "independent_test_set",
    "test_samples": int(len(y_test)),
    "accuracy": float(accuracy),
    "precision_macro": float(precision_macro),
    "recall_macro": float(recall_macro),
    "f1_macro": float(f1_macro),
    "f1_weighted": float(f1_weighted),
    "classification_report": report,
    "confusion_matrix": cm.tolist(),
    "class_order": labels
}

with open(
    os.path.join(OUTPUT_DIR, "final_metrics.json"), "w"
) as f:
    json.dump(metrics, f, indent=4)

pd.DataFrame(
    cm, index=labels, columns=labels
).to_csv(
    os.path.join(OUTPUT_DIR, "confusion_matrix.csv")
)

pd.DataFrame({
    "actual": y_test,
    "predicted": y_pred
}).to_csv(
    os.path.join(OUTPUT_DIR, "test_predictions.csv"),
    index=False
)

print("\n========== SAVED RESULTS ==========")
print(OUTPUT_DIR + "/final_metrics.json")
print(OUTPUT_DIR + "/confusion_matrix.csv")
print(OUTPUT_DIR + "/test_predictions.csv")

print("\nEvaluation completed successfully.")
