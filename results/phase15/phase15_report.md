# Phase 15 – Comparaison et sélection des modèles

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
| Accuracy moyenne | 99.996075% | 99.941412% |
| Precision macro moyenne | 99.998650% | 99.642400% |
| Recall macro moyen | 99.913812% | 99.183074% |
| F1-macro moyen | 99.956189% | 99.401250% |
| Écart-type F1-macro | 0.028712% | 0.644732% |
| Temps moyen d'entraînement | 64.41 s | 65.17 s |

Le Random Forest avec Destination Port obtient un F1-macro moyen
supérieur sur les cinq folds.

## 4. Analyse par classe

Les métriques suivantes sont calculées à partir de la matrice de confusion
agrégée des prédictions Out-of-Fold du modèle avec Destination Port.

| Classe | Support | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| BENIGN | 345659 | 99.995950% | 100.000000% | 99.997975% |
| FTP-Patator | 6350 | 100.000000% | 99.889764% | 99.944851% |
| SSH-Patator | 4718 | 100.000000% | 99.851632% | 99.925761% |

La matrice de confusion contient 14 erreurs sur 356727 prédictions
Out-of-Fold, soit un taux d'erreur global de 0.00392 %.

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
