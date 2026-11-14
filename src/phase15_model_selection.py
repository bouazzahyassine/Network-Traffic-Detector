import json
from pathlib import Path

import pandas as pd

BASE = Path(".")
OUT = BASE / "results/phase15"
OUT.mkdir(parents=True, exist_ok=True)

phase14 = OUT.parent / "robustness/phase14_4_corrected"
phase143 = OUT.parent / "robustness"

# 1. Charger les résultats de validation déjà enregistrés
summary14 = json.loads(
    (phase14 / "phase14_4_summary.json").read_text()
)
folds = pd.read_csv(phase14 / "phase14_4_fold_metrics.csv")
comparison = pd.read_csv(phase14 / "phase14_4_comparison.csv")

summary143 = json.loads(
    (phase143 / "phase14_3_summary.json").read_text()
)

phase12 = json.loads(
    (OUT.parent / "optimization/phase12_comparison.json").read_text()
)

phase13 = json.loads(
    (OUT.parent / "error_analysis/phase13_summary.json").read_text()
)

# 2. Vérifier l'intégrité du protocole
integrity = summary14.get("integrity", {})

assert integrity.get("all_folds_group_separated") is True
assert integrity.get("same_folds_for_both_models") is True
assert integrity.get("test_set_used") is False
assert integrity.get("preprocessing_inside_each_fold") is True
assert len(folds) == 5

assert (folds["overlapping_groups"] == 0).all()

# 3. Construire le tableau comparatif RF
metrics = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
]

rows = []

for model, prefix in [
    ("Random Forest avec Destination Port", "with_port"),
    ("Random Forest sans Destination Port", "without_port"),
]:
    row = {
        "model": model,
        "mean_training_time_seconds":
            folds[f"{prefix}_training_time"].mean(),
        "std_training_time_seconds":
            folds[f"{prefix}_training_time"].std(ddof=1),
    }

    for metric in metrics:
        col = f"{prefix}_{metric}"
        row[f"{metric}_mean"] = folds[col].mean()
        row[f"{metric}_std"] = folds[col].std(ddof=1)
        row[f"{metric}_min"] = folds[col].min()
        row[f"{metric}_max"] = folds[col].max()

    rows.append(row)

rf_table = pd.DataFrame(rows)
rf_table.to_csv(OUT / "phase15_model_comparison.csv", index=False)

# 4. Comparaison appariée fold par fold
paired = pd.DataFrame({
    "fold": folds["fold"],
    "f1_with_port": folds["with_port_f1_macro"],
    "f1_without_port": folds["without_port_f1_macro"],
    "f1_difference_without_minus_with": (
        folds["without_port_f1_macro"]
        - folds["with_port_f1_macro"]
    ),
    "recall_with_port": folds["with_port_recall_macro"],
    "recall_without_port": folds["without_port_recall_macro"],
    "training_time_with_port": folds["with_port_training_time"],
    "training_time_without_port": folds["without_port_training_time"],
})

paired.to_csv(OUT / "phase15_paired_fold_analysis.csv", index=False)

# 5. Préparer la comparaison historique Isolation Forest
if_summary = phase13["isolation_forest"]

isolation_forest = {
    "evaluation_source": "Phase 13 - ancienne évaluation",
    "evaluation_samples": phase13["dataset"]["test_samples"],
    "task": "Détection binaire BENIGN / ATTACK",
    "accuracy": if_summary["accuracy"],
    "precision_attack": if_summary["precision_attack"],
    "recall_attack": if_summary["recall_attack"],
    "f1_attack": if_summary["f1_attack"],
    "false_positives": if_summary["false_positives"],
    "false_negatives": if_summary["false_negatives"],
    "average_precision": if_summary["average_precision"],
    "threshold": if_summary["threshold"],
}

# 6. Résumé général
best_fold = int(
    paired.loc[
        paired["f1_with_port"].idxmax(), "fold"
    ]
)

summary = {
    "phase": "15",
    "title": "Model Comparison and Selection",
    "dataset": "CIC-IDS2017 Tuesday Working Hours",
    "random_forest_validation": {
        "protocol": "5-fold StratifiedGroupKFold",
        "same_folds_for_both_models": True,
        "all_folds_group_separated": True,
        "test_set_used_in_phase14_4": False,
        "with_destination_port": {
            "f1_macro_mean": float(
                folds["with_port_f1_macro"].mean()
            ),
            "f1_macro_std": float(
                folds["with_port_f1_macro"].std(ddof=1)
            ),
            "recall_macro_mean": float(
                folds["with_port_recall_macro"].mean()
            ),
            "mean_training_time_seconds": float(
                folds["with_port_training_time"].mean()
            ),
        },
        "without_destination_port": {
            "f1_macro_mean": float(
                folds["without_port_f1_macro"].mean()
            ),
            "f1_macro_std": float(
                folds["without_port_f1_macro"].std(ddof=1)
            ),
            "recall_macro_mean": float(
                folds["without_port_recall_macro"].mean()
            ),
            "mean_training_time_seconds": float(
                folds["without_port_training_time"].mean()
            ),
        },
        "paired_folds": paired.to_dict(orient="records"),
        "best_with_port_fold": best_fold,
    },
    "phase14_3_reference": {
        "f1_macro_mean": summary143["mean_metrics"]["f1_macro"],
        "recall_macro_mean": summary143["mean_metrics"]["recall_macro"],
    },
    "isolation_forest_historical_evaluation": isolation_forest,
    "phase12_historical_results": phase12,
    "methodological_notes": [
        "Random Forest is a 3-class supervised classifier.",
        "Isolation Forest is evaluated as a binary anomaly detector.",
        "Their F1 scores are not directly equivalent.",
        "Phase 13 already used the test set for model evaluation.",
        "Do not describe the test set as completely unseen.",
        "Do not use historical test metrics to tune model parameters.",
    ],
    "selection": {
        "provisional_preference":
            "Random Forest with Destination Port",
        "reason":
            "Higher mean group-validation F1-macro and recall-macro "
            "than the paired model without Destination Port.",
        "caveat":
            "Destination Port importance and possible dataset-specific "
            "patterns must be discussed as limitations."
    }
}

(OUT / "phase15_summary.json").write_text(
    json.dumps(summary, indent=2)
)

# 7. Générer un rapport Markdown
with_port = summary["random_forest_validation"]["with_destination_port"]
without_port = summary["random_forest_validation"]["without_destination_port"]

report = f"""# Phase 15 – Model Comparison and Selection

## 1. Objectif

Comparer les modèles à partir des résultats de validation groupée déjà produits,
sans réentraîner les modèles et sans réutiliser le jeu de test.

## 2. Protocole

- Dataset : CIC-IDS2017 Tuesday Working Hours
- Validation : StratifiedGroupKFold
- Nombre de folds : 5
- Même folds pour les deux variantes Random Forest : oui
- Groupes dupliqués entre Fit et Validation : aucun
- Jeu de test utilisé pendant la Phase 14.4 : non

## 3. Comparaison Random Forest

| Métrique | Avec Destination Port | Sans Destination Port |
|---|---:|---:|
| F1-macro moyen | {with_port['f1_macro_mean']:.6%} | {without_port['f1_macro_mean']:.6%} |
| F1-macro écart-type | {with_port['f1_macro_std']:.6%} | {without_port['f1_macro_std']:.6%} |
| Recall-macro moyen | {with_port['recall_macro_mean']:.6%} | {without_port['recall_macro_mean']:.6%} |
| Temps moyen d'entraînement | {with_port['mean_training_time_seconds']:.2f} s | {without_port['mean_training_time_seconds']:.2f} s |

## 4. Isolation Forest

Les résultats Isolation Forest proviennent de l'ancienne évaluation de la Phase 13.
Il s'agit d'une tâche binaire BENIGN/ATTACK, contrairement à la classification
multiclasse du Random Forest. Les scores F1 ne sont donc pas directement comparables.

- Accuracy : {if_summary['accuracy']:.4%}
- Precision attaque : {if_summary['precision_attack']:.4%}
- Recall attaque : {if_summary['recall_attack']:.4%}
- F1 attaque : {if_summary['f1_attack']:.4%}
- Faux positifs : {if_summary['false_positives']}
- Faux négatifs : {if_summary['false_negatives']}

## 5. Décision provisoire

Le Random Forest avec Destination Port est retenu comme modèle préféré à ce stade,
car il obtient les meilleurs résultats moyens sur les mêmes folds groupés.

Cette décision reste conditionnée à l'analyse des limites du dataset et de
l'importance de Destination Port. Une excellente performance ne prouve pas,
à elle seule, une généralisation à d'autres réseaux ou environnements.

## 6. Limitation concernant le jeu de test

La Phase 13 indique que le jeu de test de 89 182 observations a déjà été utilisé
pour évaluer les modèles. Il ne peut donc plus être présenté comme un jeu de test
totalement inédit. Aucun nouvel ajustement ne doit être réalisé sur la base de
ces résultats historiques.

## 7. Fichiers produits

- phase15_model_comparison.csv
- phase15_paired_fold_analysis.csv
- phase15_summary.json
- phase15_report.md
"""

(OUT / "phase15_report.md").write_text(report)

print("\n========== PHASE 15.1 TERMINÉE ==========")
print("\nComparaison Random Forest :")
print(rf_table.to_string(index=False))

print("\nAnalyse fold par fold :")
print(paired.to_string(index=False))

print("\nIsolation Forest (évaluation historique) :")
print(json.dumps(isolation_forest, indent=2))

print("\nFichiers créés dans :", OUT)
for path in sorted(OUT.iterdir()):
    print("-", path)
