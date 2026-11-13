
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================

FILE = "data/processed/Tuesday_clean.csv"
OUTPUT = "results/eda"

os.makedirs(OUTPUT, exist_ok=True)

sns.set_theme(style="whitegrid")

print("=" * 60)
print("PHASE 9 - FEATURE ANALYSIS AND ADVANCED EDA")
print("=" * 60)

# ==========================================
# 1. LOAD DATASET
# ==========================================

print("\n[1] Loading dataset...")

df = pd.read_csv(FILE, low_memory=False)

df.columns = df.columns.str.strip()

print("Dataset shape:", df.shape)
print("Classes:", df["Label"].nunique())

# ==========================================
# 2. CLASS DISTRIBUTION
# ==========================================

print("\n[2] Analyzing class distribution...")

class_counts = df["Label"].value_counts()

class_percentages = (
    df["Label"].value_counts(normalize=True) * 100
)

class_distribution = pd.DataFrame({
    "Count": class_counts,
    "Percentage": class_percentages
})

print(class_distribution)

class_distribution.to_csv(
    f"{OUTPUT}/class_distribution.csv"
)

plt.figure(figsize=(9, 5))

sns.barplot(
    x=class_counts.index,
    y=class_counts.values
)

plt.title("Network Traffic Class Distribution")
plt.xlabel("Traffic Class")
plt.ylabel("Number of Flows")
plt.xticks(rotation=15)

plt.tight_layout()
plt.savefig(f"{OUTPUT}/class_distribution.png", dpi=300)
plt.close()

# ==========================================
# 3. NUMERICAL FEATURES
# ==========================================

print("\n[3] Analyzing numerical features...")

numeric_df = df.select_dtypes(include=[np.number])

features = numeric_df.columns.tolist()

print("Number of numerical features:", len(features))

statistics = numeric_df.describe().T

statistics["missing_values"] = numeric_df.isna().sum()

statistics.to_csv(
    f"{OUTPUT}/feature_statistics.csv"
)

print(statistics.head(10))

# ==========================================
# 4. CONSTANT FEATURES
# ==========================================

print("\n[4] Detecting constant features...")

constant_features = [
    col for col in features
    if numeric_df[col].nunique(dropna=False) <= 1
]

print("Constant features:", constant_features)

pd.DataFrame({
    "Constant_Features": constant_features
}).to_csv(
    f"{OUTPUT}/constant_features.csv",
    index=False
)

# ==========================================
# 5. DUPLICATED COLUMNS
# ==========================================

print("\n[5] Detecting identical columns...")

duplicate_columns = []

for i, col1 in enumerate(df.columns):
    for col2 in df.columns[i + 1:]:
        if df[col1].equals(df[col2]):
            duplicate_columns.append((col1, col2))

print("Identical column pairs:", duplicate_columns)

with open(
    f"{OUTPUT}/duplicate_columns.txt",
    "w"
) as file:

    for col1, col2 in duplicate_columns:
        file.write(f"{col1} == {col2}\n")

# ==========================================
# 6. FEATURE HISTOGRAMS
# ==========================================

print("\n[6] Generating feature histograms...")

selected_features = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Packet Length Mean",
    "SYN Flag Count",
    "Average Packet Size"
]

available_features = [
    col for col in selected_features
    if col in numeric_df.columns
]

# Balanced sample for visualization
sample_parts = []

for label, group in df.groupby("Label"):
    sample_parts.append(
        group.sample(
            n=min(len(group), 5000),
            random_state=42
        )
    )

sample_df = pd.concat(sample_parts)

for col in available_features:

    plt.figure(figsize=(9, 5))

    values = pd.to_numeric(
        sample_df[col],
        errors="coerce"
    ).dropna()

    sns.histplot(
        values,
        bins=50,
        kde=True
    )

    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")

    plt.tight_layout()

    filename = col.replace("/", "_").replace(" ", "_")

    plt.savefig(
        f"{OUTPUT}/hist_{filename}.png",
        dpi=250
    )

    plt.close()

# ==========================================
# 7. BOXPLOTS BY CLASS
# ==========================================

print("\n[7] Generating boxplots...")

for col in available_features:

    plot_df = sample_df[["Label", col]].copy()

    plot_df[col] = pd.to_numeric(
        plot_df[col],
        errors="coerce"
    )

    plot_df = plot_df.dropna()

    plt.figure(figsize=(10, 5))

    sns.boxplot(
        data=plot_df,
        x="Label",
        y=col
    )

    plt.title(f"{col} by Traffic Class")
    plt.xticks(rotation=15)

    plt.tight_layout()

    filename = col.replace("/", "_").replace(" ", "_")

    plt.savefig(
        f"{OUTPUT}/boxplot_{filename}.png",
        dpi=250
    )

    plt.close()

# ==========================================
# 8. CORRELATION MATRIX
# ==========================================

print("\n[8] Computing correlation matrix...")

correlation = numeric_df.corr()

correlation.to_csv(
    f"{OUTPUT}/correlation_matrix.csv"
)

plt.figure(figsize=(22, 18))

sns.heatmap(
    correlation,
    cmap="coolwarm",
    center=0,
    linewidths=0.1,
    xticklabels=True,
    yticklabels=True
)

plt.title("Network Features Correlation Matrix")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT}/correlation_matrix.png",
    dpi=300
)

plt.close()

# ==========================================
# 9. HIGHLY CORRELATED FEATURES
# ==========================================

print("\n[9] Detecting highly correlated features...")

upper = correlation.abs().where(
    np.triu(
        np.ones(correlation.shape),
        k=1
    ).astype(bool)
)

high_corr = (
    upper.stack()
    .reset_index()
)

high_corr.columns = [
    "Feature_1",
    "Feature_2",
    "Correlation"
]

high_corr = high_corr[
    high_corr["Correlation"] >= 0.95
].sort_values(
    by="Correlation",
    ascending=False
)

print(high_corr.head(30))

high_corr.to_csv(
    f"{OUTPUT}/high_correlation_pairs.csv",
    index=False
)

# ==========================================
# 10. FINAL SUMMARY
# ==========================================

print("\n" + "=" * 60)
print("PHASE 9 - EDA SUMMARY")
print("=" * 60)

print("Total rows:", len(df))
print("Total features:", len(features))

print("Constant features:", len(constant_features))

print("Identical column pairs:", len(duplicate_columns))

print("Highly correlated pairs:", len(high_corr))

print("\nGenerated files:")

for filename in sorted(os.listdir(OUTPUT)):
    print("-", filename)

print("\nPHASE 9 COMPLETED SUCCESSFULLY!")
