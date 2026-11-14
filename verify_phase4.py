import json
import pandas as pd
import numpy as np
from pathlib import Path

base = Path("results/final_evaluation")

with open(base / "final_metrics.json") as f:
    metrics = json.load(f)

cm = pd.read_csv(
    base / "confusion_matrix.csv",
    index_col=0
)

pred = pd.read_csv(
    base / "test_predictions.csv"
)

print("========== PHASE 4 VERIFICATION ==========")

print("\n1. DATASET")
print("Predictions:", len(pred))
print("Missing values:", pred.isna().sum().sum())

print("\n2. COLUMNS")
print(pred.columns.tolist())

print("\n3. CONFUSION MATRIX")
print(cm)

print("\n4. MATRIX TOTAL")
print("Total:", cm.to_numpy().sum())

print("\n5. METRICS")
print(json.dumps(metrics, indent=4))

print("\n6. PREDICTION ERRORS")

errors = (pred["actual"] != pred["predicted"]).sum()

print("Incorrect:", errors)
print("Correct:", len(pred) - errors)

accuracy = (pred["actual"] == pred["predicted"]).mean()

print("Calculated Accuracy:", accuracy * 100, "%")

print("\n========== END ==========")
