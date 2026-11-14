import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE = "results"
FIGURES = os.path.join(BASE, "figures")
os.makedirs(FIGURES, exist_ok=True)

print("=" * 65)
print("GENERATING FINAL PROJECT GRAPHS")
print("=" * 65)

# -------------------------------------------------
# 1. Load project results
# -------------------------------------------------
df = pd.read_csv("data/processed/Tuesday_clean.csv")

metrics_path = os.path.join(
    BASE, "final_evaluation", "final_metrics.json"
)

with open(metrics_path, "r") as f:
    metrics = json.load(f)

cm = pd.read_csv(
    os.path.join(
        BASE, "final_evaluation", "confusion_matrix.csv"
    ),
    index_col=0
)

labels = ["BENIGN", "FTP-Patator", "SSH-Patator"]

# -------------------------------------------------
# GRAPH 1: Dataset class distribution
# -------------------------------------------------
print("\n[1] Dataset class distribution...")

class_counts = df["Label"].value_counts()

plt.figure(figsize=(10, 6))
class_counts.plot(kind="bar")
plt.title("CIC-IDS2017 - Class Distribution")
plt.xlabel("Traffic class")
plt.ylabel("Number of observations")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(
    os.path.join(FIGURES, "01_class_distribution.png"),
    dpi=300
)
plt.close()

# -------------------------------------------------
# GRAPH 2: Final confusion matrix
# -------------------------------------------------
print("[2] Final confusion matrix...")

values = cm.loc[labels, labels].values

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(values)

ax.set_title("Random Forest - Final Confusion Matrix")
ax.set_xlabel("Predicted class")
ax.set_ylabel("Actual class")

ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_yticklabels(labels)

for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(
            j, i, f"{values[i, j]:,}",
            ha="center", va="center"
        )

fig.colorbar(im, ax=ax, label="Number of observations")
plt.tight_layout()
plt.savefig(
    os.path.join(FIGURES, "02_confusion_matrix.png"),
    dpi=300
)
plt.close()

# -------------------------------------------------
# GRAPH 3: Per-class metrics
# -------------------------------------------------
print("[3] Per-class performance...")

report = metrics["classification_report"]

metric_names = ["precision", "recall", "f1-score"]

class_values = {
    label: [
        report[label][metric]
        for metric in metric_names
    ]
    for label in labels
}

x = np.arange(len(metric_names))
width = 0.23

plt.figure(figsize=(10, 6))

for i, label in enumerate(labels):
    plt.bar(
        x + (i - 1) * width,
        class_values[label],
        width,
        label=label
    )

plt.title("Random Forest - Performance by Class")
plt.xlabel("Evaluation metric")
plt.ylabel("Score")
plt.xticks(x, metric_names)
plt.ylim(0.98, 1.001)
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(FIGURES, "03_class_performance.png"),
    dpi=300
)
plt.close()

# -------------------------------------------------
# GRAPH 4: Global final metrics
# -------------------------------------------------
print("[4] Global performance metrics...")

global_names = [
    "Accuracy",
    "Precision macro",
    "Recall macro",
    "F1 macro",
    "F1 weighted"
]

global_values = [
    metrics["accuracy"],
    metrics["precision_macro"],
    metrics["recall_macro"],
    metrics["f1_macro"],
    metrics["f1_weighted"]
]

plt.figure(figsize=(11, 6))
bars = plt.bar(global_names, global_values)

plt.title("Random Forest - Final Test Metrics")
plt.ylabel("Score")
plt.ylim(0.98, 1.001)
plt.xticks(rotation=15, ha="right")

for bar, value in zip(bars, global_values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value * 100:.4f}%",
        ha="center",
        va="bottom",
        fontsize=8
    )

plt.tight_layout()
plt.savefig(
    os.path.join(FIGURES, "04_global_metrics.png"),
    dpi=300
)
plt.close()

# -------------------------------------------------
# GRAPH 5: Destination Port sensitivity
# -------------------------------------------------
print("[5] Destination Port sensitivity...")

with_port = [99.996075, 99.9987, 99.9138, 99.9562]
without_port = [99.941412, 99.6424, 99.1831, 99.4013]

comparison_names = [
    "Accuracy",
    "Precision macro",
    "Recall macro",
    "F1 macro"
]

x = np.arange(len(comparison_names))
width = 0.35

plt.figure(figsize=(11, 6))

plt.bar(
    x - width / 2,
    with_port,
    width,
    label="With Destination Port"
)

plt.bar(
    x + width / 2,
    without_port,
    width,
    label="Without Destination Port"
)

plt.title("Destination Port Sensitivity Analysis - Phase 14.4")
plt.xlabel("Metric")
plt.ylabel("Score (%)")
plt.xticks(x, comparison_names)
plt.ylim(98.5, 100.05)
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(FIGURES, "05_destination_port_comparison.png"),
    dpi=300
)
plt.close()

# -------------------------------------------------
# Final verification
# -------------------------------------------------
print("\n========== GENERATED FIGURES ==========")

for filename in sorted(os.listdir(FIGURES)):
    if filename.endswith(".png"):
        path = os.path.join(FIGURES, filename)
        print(filename, "-", round(os.path.getsize(path) / 1024, 1), "KB")

print("\nAll figures generated successfully.")
print("Output directory:", FIGURES)
