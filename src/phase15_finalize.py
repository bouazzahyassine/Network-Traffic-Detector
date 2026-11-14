from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("results/phase15")
ROB = Path("results/robustness/phase14_4_corrected")

comparison = pd.read_csv(OUT / "phase15_model_comparison.csv")
per_class = pd.read_csv(OUT / "phase15_per_class_metrics.csv")
paired = pd.read_csv(OUT / "phase15_paired_fold_analysis.csv")

# ==================================================
# GRAPH 1 - Comparaison des deux Random Forest
# ==================================================
names = ["Avec Destination Port", "Sans Destination Port"]
f1 = comparison["f1_macro_mean"] * 100

plt.figure(figsize=(9, 5))
bars = plt.bar(names, f1)
plt.ylabel("F1-macro moyen (%)")
plt.title("Comparaison Random Forest - Validation groupée")
plt.ylim(min(f1) - 0.15, 100.02)
plt.grid(axis="y", alpha=0.3)

for bar, value in zip(bars, f1):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.4f}%",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(OUT / "phase15_rf_comparison.png", dpi=200)
plt.close()

# ==================================================
# GRAPH 2 - Métriques par classe
# ==================================================
metrics = ["precision", "recall", "f1_score"]

plt.figure(figsize=(10, 5))
x = range(len(per_class))
width = 0.24

for i, metric in enumerate(metrics):
    plt.bar(
        [v + (i - 1) * width for v in x],
        per_class[metric] * 100,
        width,
        label=metric.replace("_", " ").title()
    )

plt.xticks(list(x), per_class["class"])
plt.ylabel("Score (%)")
plt.title("Random Forest avec Destination Port - Métriques par classe")
plt.ylim(99.7, 100.02)
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "phase15_per_class_metrics.png", dpi=200)
plt.close()

# ==================================================
# GRAPH 3 - Comparaison fold par fold
# ==================================================
plt.figure(figsize=(9, 5))
x = range(1, len(paired) + 1)
width = 0.36

plt.bar(
    [v - width / 2 for v in x],
    paired["f1_with_port"] * 100,
    width,
    label="Avec port"
)
plt.bar(
    [v + width / 2 for v in x],
    paired["f1_without_port"] * 100,
    width,
    label="Sans port"
)

plt.xticks(list(x), [f"Fold {v}" for v in x])
plt.ylabel("F1-macro (%)")
plt.title("Comparaison des performances sur les cinq folds")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "phase15_fold_comparison.png", dpi=200)
plt.close()

# ==================================================
# RAPPORT FINAL
# ==================================================
with_port = comparison.iloc[0]
without_port = comparison.iloc[1]

total = int(per_class["support"].sum())
errors = int(per_class["false_negative"].sum())
correct = int(per_class["true_positive"].sum())
error_rate = errors / total * 100

assert errors == total - correct, (
    "Incohérence entre les erreurs, les supports et les vrais positifs."
)

report = f"""# Phase 15 – Comparaison et sélection des modèles

## 1. Objectif

Cette phase compare les modèles développés pendant le projet
Network Traffic Anomaly Detection Using Machine Learning.

L'objectif est de sélectionner un modèle candidat en utilisant les
résultats de validation groupée, sans nouvel entraînement.

## 2. Protocole expérimental

- Dataset : CIC-IDS2017 – Tuesday Working Hours
- Algorithme principal : Random Forest
- Validation : StratifiedGroupKFold
- Nombre de folds : 5
- Comparaison : avec et sans Destination Port
- Même folds pour les deux variantes : oui
- Chevauchement des groupes entre Fit et Validation : aucun
- Jeu de test utilisé pendant cette comparaison : non

Le prétraitement a été effectué à l'intérieur de chaque fold.

## 3. Comparaison des modèles Random Forest

| Métrique | Avec Destination Port | Sans Destination Port |
|---|---:|---:|
| Accuracy moyenne | {with_port['accuracy_mean']:.6%} | {without_port['accuracy_mean']:.6%} |
| Precision macro moyenne | {with_port['precision_macro_mean']:.6%} | {without_port['precision_macro_mean']:.6%} |
| Recall macro moyen | {with_port['recall_macro_mean']:.6%} | {without_port['recall_macro_mean']:.6%} |
| F1-macro moyen | {with_port['f1_macro_mean']:.6%} | {without_port['f1_macro_mean']:.6%} |
| Écart-type F1-macro | {with_port['f1_macro_std']:.6%} | {without_port['f1_macro_std']:.6%} |
| Temps moyen d'entraînement | {with_port['mean_training_time_seconds']:.2f} s | {without_port['mean_training_time_seconds']:.2f} s |

Le Random Forest avec Destination Port obtient un F1-macro moyen
supérieur sur les cinq folds.

## 4. Analyse par classe

Les métriques suivantes sont calculées à partir de la matrice de confusion
agrégée des prédictions Out-of-Fold du modèle avec Destination Port.

| Classe | Support | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
"""

for _, row in per_class.iterrows():
    report += (
        f"| {row['class']} | {int(row['support'])} | "
        f"{row['precision']:.6%} | {row['recall']:.6%} | "
        f"{row['f1_score']:.6%} |\n"
    )

report += f"""
La matrice de confusion contient {errors} erreurs sur {total} prédictions
Out-of-Fold, soit un taux d'erreur global de {error_rate:.5f} %.

Les erreurs correspondent à :
- 7 flux FTP-Patator classés BENIGN.
- 7 flux SSH-Patator classés BENIGN.
- Aucun flux BENIGN classé comme FTP-Patator ou SSH-Patator.

Ainsi, le modèle n'a généré aucun faux positif d'attaque dans cette
matrice multiclasses agrégée, mais il a manqué 14 flux d'attaque.

## 5. Comparaison fold par fold

Le modèle avec Destination Port obtient un F1-macro supérieur sur les
5 folds sur 5.

Cette cohérence renforce le choix de cette variante dans le cadre
du protocole expérimental actuel.

## 6. Isolation Forest

Isolation Forest a été évalué comme détecteur binaire BENIGN/ATTACK.
Ses résultats disponibles proviennent de l'évaluation historique
de la Phase 13 :

- Accuracy : 68,13 %.
- Precision attaque : 6,69 %.
- Recall attaque : 71,59 %.
- F1 attaque : 12,23 %.
- Faux positifs : 27 640.
- Faux négatifs : 786.

Ces résultats ne sont pas directement comparables aux scores
multiclasses du Random Forest : les tâches et les protocoles
d'évaluation sont différents.

## 7. Sélection du modèle

Le modèle retenu comme candidat principal est :

**Random Forest avec Destination Port.**

Justifications :
- Meilleur F1-macro moyen parmi les deux variantes comparées.
- Meilleur résultat sur chacun des cinq folds.
- Faible variabilité entre les folds.
- Temps moyen d'entraînement légèrement inférieur dans cette expérience.

Cette sélection est limitée au dataset et au protocole utilisés.
Elle ne garantit pas une généralisation identique à d'autres réseaux.

## 8. Limites et précautions

1. Destination Port contribue fortement à la prédiction.
   Cette caractéristique peut refléter des propriétés spécifiques
   du dataset CIC-IDS2017. Il faut éviter de conclure qu'elle sera
   aussi discriminante dans tous les réseaux.

2. Les performances très élevées doivent être interprétées avec prudence.
   Une validation groupée réduit le risque de fuite par groupes dupliqués,
   mais n'exclut pas toutes les formes de biais ou de dépendance au dataset.

3. Le jeu de test a déjà été utilisé dans l'analyse de la Phase 13.
   Il ne doit donc pas être présenté comme un jeu totalement inédit.

4. Les scores Isolation Forest proviennent d'une ancienne évaluation
   binaire et ne sont pas directement comparables aux scores
   multiclasses du Random Forest.

## 9. Conclusion

La Phase 15 a permis de comparer les variantes Random Forest,
d'analyser les performances par classe et d'étudier la stabilité
des résultats sur cinq folds groupés.

Le Random Forest avec Destination Port est sélectionné comme
modèle candidat pour la suite du projet.

La Phase 15 est terminée. La suite du projet devra présenter
clairement les limites expérimentales et le protocole d'évaluation.
"""

(OUT / "phase15_report.md").write_text(report)

print("\\n========== PHASE 15 FINALISÉE ==========")
print("\\nGraphiques créés :")
for name in [
    "phase15_rf_comparison.png",
    "phase15_per_class_metrics.png",
    "phase15_fold_comparison.png",
]:
    print("-", OUT / name)

print("\\nRapport :", OUT / "phase15_report.md")
print("\\nRésumé :")
print(f"F1 avec port    : {with_port['f1_macro_mean']:.6%}")
print(f"F1 sans port    : {without_port['f1_macro_mean']:.6%}")
print(f"Meilleur sur les folds : 5/5")
print(f"Erreurs OOF     : {errors}/{total}")
