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
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# PHASE 14.3 - GROUP-AWARE CROSS-VALIDATION FOR RANDOM FOREST
# ============================================================

print("=" * 70)
print("PHASE 14.3 - GROUP-AWARE CROSS-VALIDATION FOR RANDOM FOREST")
print("=" * 70)

RANDOM_STATE = 42
N_SPLITS = 5

DATA_FILE = "data/processed/Tuesday_clean.csv"
PREPARED_FILE = "data/prepared/prepared_dataset.joblib"

RESULTS_DIR = "results/robustness"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

print("\n[1] Loading datasets...")

df = pd.read_csv(DATA_FILE)

prepared = joblib.load(PREPARED_FILE)

train_indices = np.asarray(prepared["train_indices"])
y_train = np.asarray(prepared["y_train"]).astype(str)

del prepared
gc.collect()

train_df = df.iloc[train_indices]

if len(train_df) != len(y_train):
    raise ValueError("Train rows and y_train have different lengths.")

raw_labels = train_df["Label"].astype(str).to_numpy()

if not np.array_equal(raw_labels, y_train):
    raise ValueError("Labels do not match the prepared dataset.")

X_raw_train = train_df.drop(columns=["Label"])

for col in X_raw_train.columns:
    X_raw_train[col] = pd.to_numeric(
        X_raw_train[col], errors="coerce"
    )

X_raw_train = X_raw_train.replace([np.inf, -np.inf], np.nan)

del df, train_df, raw_labels, train_indices
gc.collect()

print("Training rows:", len(X_raw_train))
print("Raw features:", X_raw_train.shape[1])
print("Classes:", np.unique(y_train))

# ------------------------------------------------------------
# 2. RECONSTRUCT DUPLICATE GROUPS
# ------------------------------------------------------------

print("\n[2] Reconstructing duplicate groups...")

groups_train = pd.util.hash_pandas_object(
    X_raw_train,
    index=False
).to_numpy(dtype="uint64")

print("Unique training groups:", len(np.unique(groups_train)))

# Convert numeric features to float32 after group reconstruction.
# This reduces memory without changing the previously calculated groups.
X_raw_train = X_raw_train.astype(np.float32, copy=False)

gc.collect()

print("Feature dtype:", X_raw_train.dtypes.unique())

# ------------------------------------------------------------
# 3. INITIALIZE GROUP-AWARE CROSS-VALIDATION
# ------------------------------------------------------------

print("\n[3] Creating 5 group-aware folds...")

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

fold_results = []
all_true = []
all_pred = []

classes = np.unique(y_train)

# ------------------------------------------------------------
# 4. TRAIN AND EVALUATE EACH FOLD
# ------------------------------------------------------------

for fold_number, (fit_pos, val_pos) in enumerate(
    cv.split(X_raw_train, y_train, groups_train),
    start=1
):

    print("\n" + "=" * 65)
    print(f"FOLD {fold_number}/{N_SPLITS}")
    print("=" * 65)

    X_fit_raw = X_raw_train.iloc[fit_pos]
    X_val_raw = X_raw_train.iloc[val_pos]

    y_fit = y_train[fit_pos]
    y_val = y_train[val_pos]

    groups_fit = groups_train[fit_pos]
    groups_val = groups_train[val_pos]

    # --------------------------------------------------------
    # 4.1 VERIFY GROUP SEPARATION
    # --------------------------------------------------------

    overlap = np.intersect1d(
        np.unique(groups_fit),
        np.unique(groups_val)
    )

    print("Fit rows:", len(fit_pos))
    print("Validation rows:", len(val_pos))
    print("Fit unique groups:", len(np.unique(groups_fit)))
    print("Validation unique groups:", len(np.unique(groups_val)))
    print("Overlapping groups:", len(overlap))

    if len(overlap) != 0:
        raise RuntimeError(
            f"Data leakage detected in fold {fold_number}!"
        )

    # --------------------------------------------------------
    # 4.2 BUILD PREPROCESSING + MODEL PIPELINE
    # --------------------------------------------------------

    pipeline = Pipeline([
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

    # --------------------------------------------------------
    # 4.3 TRAIN
    # --------------------------------------------------------

    print("\nTraining Random Forest...")

    start_time = time.time()

    pipeline.fit(X_fit_raw, y_fit)

    training_time = time.time() - start_time

    print(f"Training time: {training_time:.2f} seconds")

    retained_features = pipeline.named_steps[
        "variance"
    ].get_support().sum()

    print("Retained features:", retained_features)

    # --------------------------------------------------------
    # 4.4 PREDICT VALIDATION
    # --------------------------------------------------------

    y_pred = pipeline.predict(X_val_raw)

    accuracy = accuracy_score(y_val, y_pred)

    precision = precision_score(
        y_val, y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_val, y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_val, y_pred,
        average="macro",
        zero_division=0
    )

    print("\nFOLD METRICS")
    print(f"Accuracy:  {accuracy:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall:    {recall:.6f}")
    print(f"F1-score:  {f1:.6f}")

    print("\nClassification Report:")
    print(classification_report(
        y_val,
        y_pred,
        zero_division=0
    ))

    fold_results.append({
        "fold": fold_number,
        "fit_rows": int(len(fit_pos)),
        "validation_rows": int(len(val_pos)),
        "fit_unique_groups": int(len(np.unique(groups_fit))),
        "validation_unique_groups": int(len(np.unique(groups_val))),
        "overlapping_groups": int(len(overlap)),
        "retained_features": int(retained_features),
        "training_time_seconds": round(training_time, 3),
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1)
    })

    all_true.extend(y_val.tolist())
    all_pred.extend(y_pred.tolist())

    # --------------------------------------------------------
    # 4.5 RELEASE MEMORY AFTER EACH FOLD
    # --------------------------------------------------------

    del pipeline
    del X_fit_raw, X_val_raw
    del y_fit, y_val, y_pred
    del groups_fit, groups_val, overlap
    del fit_pos, val_pos

    gc.collect()

    print(f"Memory cleanup completed after Fold {fold_number}.")

# ------------------------------------------------------------
# 5. AGGREGATE CROSS-VALIDATION RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("[5] AGGREGATE CROSS-VALIDATION RESULTS")
print("=" * 70)

metrics_df = pd.DataFrame(fold_results)

metric_columns = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro"
]

mean_metrics = metrics_df[metric_columns].mean()
std_metrics = metrics_df[metric_columns].std(ddof=1)

print("\nMean metrics:")
print(mean_metrics)

print("\nStandard deviation:")
print(std_metrics)

print("\nFold results:")
print(metrics_df.to_string(index=False))

# ------------------------------------------------------------
# 6. OUT-OF-FOLD CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\n[6] Generating aggregate classification report...")

aggregate_report = classification_report(
    all_true,
    all_pred,
    labels=classes,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(aggregate_report).transpose()

aggregate_cm = confusion_matrix(
    all_true,
    all_pred,
    labels=classes
)

cm_df = pd.DataFrame(
    aggregate_cm,
    index=[f"Actual_{c}" for c in classes],
    columns=[f"Predicted_{c}" for c in classes]
)

print("\nAggregate confusion matrix:")
print(cm_df)

# ------------------------------------------------------------
# 7. SAVE RESULTS
# ------------------------------------------------------------

print("\n[7] Saving results...")

metrics_df.to_csv(
    f"{RESULTS_DIR}/phase14_3_fold_metrics.csv",
    index=False
)

report_df.to_csv(
    f"{RESULTS_DIR}/phase14_3_classification_report.csv"
)

cm_df.to_csv(
    f"{RESULTS_DIR}/phase14_3_oof_confusion_matrix.csv"
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=aggregate_cm,
    display_labels=classes
)

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title("Phase 14.3 - Random Forest Group-Aware CV")
plt.tight_layout()

plt.savefig(
    f"{RESULTS_DIR}/phase14_3_oof_confusion_matrix.png",
    dpi=300
)

plt.close()

summary = {
    "phase": "14.3",
    "title": "Group-Aware Cross-Validation for Random Forest",
    "dataset": "CIC-IDS2017 Tuesday Working Hours",
    "algorithm": "Random Forest",
    "split_method": "StratifiedGroupKFold",
    "n_splits": N_SPLITS,
    "random_state": RANDOM_STATE,
    "training_rows": int(len(y_train)),
    "classes": classes.tolist(),
    "folds": fold_results,
    "mean_metrics": {
        key: float(mean_metrics[key])
        for key in metric_columns
    },
    "std_metrics": {
        key: float(std_metrics[key])
        for key in metric_columns
    },
    "aggregate_confusion_matrix": {
        "labels": classes.tolist(),
        "matrix": aggregate_cm.tolist()
    },
    "integrity": {
        "all_folds_group_separated": bool(
            metrics_df["overlapping_groups"].eq(0).all()
        ),
        "test_set_used": False,
        "preprocessing_fitted_inside_each_fold": True
    }
}

with open(
    f"{RESULTS_DIR}/phase14_3_summary.json",
    "w"
) as f:
    json.dump(summary, f, indent=4)

# ------------------------------------------------------------
# 8. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PHASE 14.3 COMPLETED")
print("=" * 70)

print("\nMean Accuracy:", round(mean_metrics["accuracy"], 6))
print("Mean Precision:", round(mean_metrics["precision_macro"], 6))
print("Mean Recall:", round(mean_metrics["recall_macro"], 6))
print("Mean F1-score:", round(mean_metrics["f1_macro"], 6))

print("\nGroup separation:",
      "PASSED" if summary["integrity"][
          "all_folds_group_separated"
      ] else "FAILED")

print("\nTest set used: NO")
print("Preprocessing leakage: PREVENTED")

print("\nSaved files:")
print(f"{RESULTS_DIR}/phase14_3_fold_metrics.csv")
print(f"{RESULTS_DIR}/phase14_3_classification_report.csv")
print(f"{RESULTS_DIR}/phase14_3_oof_confusion_matrix.csv")
print(f"{RESULTS_DIR}/phase14_3_oof_confusion_matrix.png")
print(f"{RESULTS_DIR}/phase14_3_summary.json")
