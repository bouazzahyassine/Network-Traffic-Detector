import gc
import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# PHASE 14.4 - DESTINATION PORT SENSITIVITY ANALYSIS
# ============================================================

print("=" * 75)
print("PHASE 14.4 - DESTINATION PORT SENSITIVITY ANALYSIS")
print("=" * 75)

RANDOM_STATE = 42
N_SPLITS = 5

DATA_FILE = "data/processed/Tuesday_clean.csv"
PREPARED_FILE = "data/prepared/prepared_dataset.joblib"
RESULTS_DIR = "results/robustness/phase14_4_corrected"

os.makedirs(RESULTS_DIR, exist_ok=True)

PORT_COLUMN = "Destination Port"

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

print("\n[1] Loading dataset...")

df = pd.read_csv(DATA_FILE)
prepared = joblib.load(PREPARED_FILE)

train_indices = np.asarray(prepared["train_indices"])
y_train = np.asarray(prepared["y_train"]).astype(str)

del prepared
gc.collect()

train_df = df.iloc[train_indices]

if len(train_df) != len(y_train):
    raise ValueError("Training rows and labels do not match.")

raw_labels = train_df["Label"].astype(str).to_numpy()

if not np.array_equal(raw_labels, y_train):
    raise ValueError("Label mismatch detected.")

X_raw_train = train_df.drop(columns=["Label"])

for col in X_raw_train.columns:
    X_raw_train[col] = pd.to_numeric(
        X_raw_train[col], errors="coerce"
    )

X_raw_train = X_raw_train.replace(
    [np.inf, -np.inf], np.nan
)

del df, train_df, raw_labels, train_indices
gc.collect()

if PORT_COLUMN not in X_raw_train.columns:
    raise ValueError("Destination Port not found!")

print("Training rows:", len(X_raw_train))
print("Original features:", X_raw_train.shape[1])
print("Destination Port:", PORT_COLUMN)

# ------------------------------------------------------------
# 2. RECONSTRUCT DUPLICATE GROUPS
# ------------------------------------------------------------

print("\n[2] Reconstructing duplicate groups WITHOUT Destination Port...")

# Build duplicate groups using only the 77 common features.
# Destination Port remains available for Model A.
X_group = X_raw_train.drop(columns=[PORT_COLUMN])

groups_train = pd.util.hash_pandas_object(
    X_group,
    index=False
).to_numpy(dtype="uint64")

print("Grouping features:", X_group.shape[1])
print("Unique groups without Destination Port:",
      len(np.unique(groups_train)))

del X_group
gc.collect()

# Same conversion strategy as Phase 14.3
X_raw_train = X_raw_train.astype(
    np.float32, copy=False
)

gc.collect()

classes = np.unique(y_train)

print("Classes:", classes)

# ------------------------------------------------------------
# 3. CREATE SAME GROUP-AWARE CV CONFIGURATION
# ------------------------------------------------------------

print("\n[3] Initializing StratifiedGroupKFold...")

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

# Store metrics and predictions
fold_results = []

all_true = []
all_pred_with = []
all_pred_without = []

# ------------------------------------------------------------
# 4. MODEL FACTORY
# ------------------------------------------------------------

def create_pipeline():

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),

        ("variance", VarianceThreshold(threshold=0.0)),

        ("model", RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1
        ))
    ])


def evaluate_model(y_true, y_pred):

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision_macro": float(
            precision_score(
                y_true, y_pred,
                average="macro",
                zero_division=0
            )
        ),
        "recall_macro": float(
            recall_score(
                y_true, y_pred,
                average="macro",
                zero_division=0
            )
        ),
        "f1_macro": float(
            f1_score(
                y_true, y_pred,
                average="macro",
                zero_division=0
            )
        )
    }


# ------------------------------------------------------------
# 5. GROUP-AWARE CROSS-VALIDATION
# ------------------------------------------------------------

print("\n[4] Starting 5-fold experiments...")

for fold_number, (fit_pos, val_pos) in enumerate(
    cv.split(X_raw_train, y_train, groups_train),
    start=1
):

    print("\n" + "=" * 75)
    print(f"FOLD {fold_number}/{N_SPLITS}")
    print("=" * 75)

    X_fit_raw = X_raw_train.iloc[fit_pos]
    X_val_raw = X_raw_train.iloc[val_pos]

    y_fit = y_train[fit_pos]
    y_val = y_train[val_pos]

    groups_fit = groups_train[fit_pos]
    groups_val = groups_train[val_pos]

    # --------------------------------------------------------
    # 5.1 VERIFY GROUP SEPARATION
    # --------------------------------------------------------

    overlap = np.intersect1d(
        np.unique(groups_fit),
        np.unique(groups_val)
    )

    print("Fit rows:", len(fit_pos))
    print("Validation rows:", len(val_pos))
    print("Overlapping groups:", len(overlap))

    if len(overlap) != 0:
        raise RuntimeError(
            f"Group leakage detected in Fold {fold_number}"
        )

    fold_data = {
        "fold": fold_number,
        "fit_rows": int(len(fit_pos)),
        "validation_rows": int(len(val_pos)),
        "fit_unique_groups": int(len(np.unique(groups_fit))),
        "validation_unique_groups": int(len(np.unique(groups_val))),
        "overlapping_groups": int(len(overlap))
    }

    # Save true labels once per fold
    all_true.extend(y_val.tolist())

    # --------------------------------------------------------
    # 5.2 MODEL A - WITH DESTINATION PORT
    # --------------------------------------------------------

    print("\n[MODEL A] WITH DESTINATION PORT")

    pipeline = create_pipeline()

    start_time = time.time()

    pipeline.fit(X_fit_raw, y_fit)

    training_time = time.time() - start_time

    retained = int(
        pipeline.named_steps["variance"].get_support().sum()
    )

    y_pred_with = pipeline.predict(X_val_raw)

    metrics_with = evaluate_model(y_val, y_pred_with)

    print("Retained features:", retained)
    print(f"Training time: {training_time:.2f}s")
    print("Accuracy:", metrics_with["accuracy"])
    print("Precision:", metrics_with["precision_macro"])
    print("Recall:", metrics_with["recall_macro"])
    print("F1-score:", metrics_with["f1_macro"])

    fold_data.update({
        "with_port_retained_features": retained,
        "with_port_training_time": round(training_time, 3)
    })

    for key, value in metrics_with.items():
        fold_data[f"with_port_{key}"] = value

    all_pred_with.extend(y_pred_with.tolist())

    del pipeline, y_pred_with
    gc.collect()

    # --------------------------------------------------------
    # 5.3 MODEL B - WITHOUT DESTINATION PORT
    # --------------------------------------------------------

    print("\n[MODEL B] WITHOUT DESTINATION PORT")

    X_fit_no_port = X_fit_raw.drop(
        columns=[PORT_COLUMN]
    )

    X_val_no_port = X_val_raw.drop(
        columns=[PORT_COLUMN]
    )

    pipeline = create_pipeline()

    start_time = time.time()

    pipeline.fit(X_fit_no_port, y_fit)

    training_time = time.time() - start_time

    retained = int(
        pipeline.named_steps["variance"].get_support().sum()
    )

    y_pred_without = pipeline.predict(X_val_no_port)

    metrics_without = evaluate_model(
        y_val, y_pred_without
    )

    print("Retained features:", retained)
    print(f"Training time: {training_time:.2f}s")
    print("Accuracy:", metrics_without["accuracy"])
    print("Precision:", metrics_without["precision_macro"])
    print("Recall:", metrics_without["recall_macro"])
    print("F1-score:", metrics_without["f1_macro"])

    fold_data.update({
        "without_port_retained_features": retained,
        "without_port_training_time": round(training_time, 3)
    })

    for key, value in metrics_without.items():
        fold_data[f"without_port_{key}"] = value

    all_pred_without.extend(y_pred_without.tolist())

    fold_results.append(fold_data)

    # --------------------------------------------------------
    # 5.4 RELEASE MEMORY
    # --------------------------------------------------------

    del pipeline
    del X_fit_raw, X_val_raw
    del X_fit_no_port, X_val_no_port
    del y_fit, y_val
    del groups_fit, groups_val, overlap
    del fit_pos, val_pos
    del y_pred_without

    gc.collect()

    print(f"\nFold {fold_number} completed.")
    print("Memory cleanup completed.")

# ------------------------------------------------------------
# 6. AGGREGATE FOLD METRICS
# ------------------------------------------------------------

print("\n[5] Aggregating results...")

metrics_df = pd.DataFrame(fold_results)

metric_names = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro"
]

summary_rows = []

for metric in metric_names:

    with_values = metrics_df[
        f"with_port_{metric}"
    ]

    without_values = metrics_df[
        f"without_port_{metric}"
    ]

    summary_rows.append({
        "metric": metric,
        "with_port_mean": with_values.mean(),
        "with_port_std": with_values.std(ddof=1),
        "without_port_mean": without_values.mean(),
        "without_port_std": without_values.std(ddof=1),
        "difference_without_minus_with":
            without_values.mean() - with_values.mean()
    })

comparison_df = pd.DataFrame(summary_rows)

print("\nFINAL COMPARISON")
print(comparison_df.to_string(index=False))

# ------------------------------------------------------------
# 7. AGGREGATE OUT-OF-FOLD CONFUSION MATRICES
# ------------------------------------------------------------

print("\n[6] Generating confusion matrices...")

cm_with = confusion_matrix(
    all_true,
    all_pred_with,
    labels=classes
)

cm_without = confusion_matrix(
    all_true,
    all_pred_without,
    labels=classes
)

cm_with_df = pd.DataFrame(
    cm_with,
    index=[f"Actual_{c}" for c in classes],
    columns=[f"Predicted_{c}" for c in classes]
)

cm_without_df = pd.DataFrame(
    cm_without,
    index=[f"Actual_{c}" for c in classes],
    columns=[f"Predicted_{c}" for c in classes]
)

print("\nWITH DESTINATION PORT:")
print(cm_with_df)

print("\nWITHOUT DESTINATION PORT:")
print(cm_without_df)

# ------------------------------------------------------------
# 8. CLASSIFICATION REPORTS
# ------------------------------------------------------------

report_with = pd.DataFrame(
    classification_report(
        all_true,
        all_pred_with,
        labels=classes,
        output_dict=True,
        zero_division=0
    )
).transpose()

report_without = pd.DataFrame(
    classification_report(
        all_true,
        all_pred_without,
        labels=classes,
        output_dict=True,
        zero_division=0
    )
).transpose()

# ------------------------------------------------------------
# 9. SAVE RESULTS
# ------------------------------------------------------------

print("\n[7] Saving results...")

metrics_df.to_csv(
    f"{RESULTS_DIR}/phase14_4_fold_metrics.csv",
    index=False
)

comparison_df.to_csv(
    f"{RESULTS_DIR}/phase14_4_comparison.csv",
    index=False
)

cm_with_df.to_csv(
    f"{RESULTS_DIR}/phase14_4_confusion_matrix_with_port.csv"
)

cm_without_df.to_csv(
    f"{RESULTS_DIR}/phase14_4_confusion_matrix_without_port.csv"
)

report_with.to_csv(
    f"{RESULTS_DIR}/phase14_4_classification_report_with_port.csv"
)

report_without.to_csv(
    f"{RESULTS_DIR}/phase14_4_classification_report_without_port.csv"
)

# ------------------------------------------------------------
# 10. COMPARISON FIGURE
# ------------------------------------------------------------

plot_metrics = ["accuracy", "recall_macro", "f1_macro"]

with_means = [
    metrics_df[f"with_port_{m}"].mean()
    for m in plot_metrics
]

without_means = [
    metrics_df[f"without_port_{m}"].mean()
    for m in plot_metrics
]

x = np.arange(len(plot_metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(
    x - width / 2,
    with_means,
    width,
    label="With Destination Port"
)

ax.bar(
    x + width / 2,
    without_means,
    width,
    label="Without Destination Port"
)

ax.set_ylabel("Score")
ax.set_title("Phase 14.4 - Destination Port Sensitivity")
ax.set_xticks(x)
ax.set_xticklabels(plot_metrics)
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    f"{RESULTS_DIR}/phase14_4_comparison.png",
    dpi=300
)

plt.close()

# ------------------------------------------------------------
# 11. SAVE SUMMARY JSON
# ------------------------------------------------------------

summary = {
    "phase": "14.4",
    "title": "Destination Port Sensitivity Analysis",
    "dataset": "CIC-IDS2017 Tuesday Working Hours",
    "split_method": "StratifiedGroupKFold",
    "n_splits": N_SPLITS,
    "random_state": RANDOM_STATE,
    "training_rows": int(len(y_train)),
    "classes": classes.tolist(),
    "features": {
        "original": int(X_raw_train.shape[1]),
        "without_destination_port":
            int(X_raw_train.shape[1] - 1)
    },
    "folds": fold_results,
    "comparison": json.loads(comparison_df.to_json(orient="records")),
    "integrity": {
        "all_folds_group_separated": bool(
            metrics_df["overlapping_groups"].eq(0).all()
        ),
        "same_folds_for_both_models": True,
        "test_set_used": False,
        "preprocessing_inside_each_fold": True
    },
    "aggregate_confusion_matrices": {
        "labels": classes.tolist(),
        "with_port": cm_with.tolist(),
        "without_port": cm_without.tolist()
    }
}

with open(
    f"{RESULTS_DIR}/phase14_4_summary.json",
    "w"
) as f:
    json.dump(summary, f, indent=4)

# ------------------------------------------------------------
# 12. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("PHASE 14.4 COMPLETED SUCCESSFULLY")
print("=" * 75)

print("\nGroup separation:",
      "PASSED" if summary["integrity"][
          "all_folds_group_separated"
      ] else "FAILED")

print("Same folds:", "YES")
print("Test set used:", "NO")
print("Preprocessing leakage:", "PREVENTED")

print("\nResults directory:", RESULTS_DIR)

print("\nGenerated files:")
for filename in sorted(os.listdir(RESULTS_DIR)):
    if filename.startswith("phase14_4"):
        print("-", filename)

print("\nPHASE 14.4 FINISHED.")
