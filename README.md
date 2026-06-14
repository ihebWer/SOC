# SOC augmenté — Moteur de priorisation d'alertes par apprentissage automatique

Code de modélisation du mémoire de fin d'études *« Vers un SOC augmenté : conception d'un moteur de décision intelligent pour l'analyse et la priorisation des alertes »* (École Hexagone, Master Sécurité des données, des réseaux et des systèmes, 2026).

Ce dépôt contient la chaîne de préparation des données, de *feature engineering* et d'entraînement du modèle de priorisation, ainsi que le modèle entraîné. Il permet de reproduire les résultats présentés au chapitre 3 du mémoire.

## Aperçu

Le système attribue à chaque alerte Wazuh un **score de priorisation** unique, produit par un modèle **LightGBM** entraîné sur le jeu de données **AIT-ADS**. Le score réordonne la file d'investigation pour concentrer les attaques réelles en tête, sans modifier le volume d'alertes émis par le SIEM.

Caractéristique clé : le vecteur de **248 caractéristiques** est calculé exclusivement à partir des **champs natifs d'une alerte Wazuh** (identifiant et niveau de règle, groupes, décodeur, agent, hôte, adresse source, tactiques/techniques MITRE, champs Suricata), sans dépendance à une étiquette de taxonomie externe. Le modèle est donc directement transposable à un flux Wazuh réel.

## Structure du dépôt

| Fichier | Rôle |
|---|---|
| `00_inventory_wazuh_json.py` | Inventaire des alertes brutes (`alerts.json`) |
| `01_normalize_wazuh_json.py` | Normalisation des alertes Wazuh |
| `02_apply_labels.py` | Étiquetage des alertes par jointure temporelle avec la vérité terrain AIT-ADS |
| `03_build_wazuh_native_features.py` | Construction des caractéristiques (passe principale, fenêtres glissantes) |
| `03b_add_rule_context_features.py` | Ajout des caractéristiques de contexte de règle |
| `03c_add_cracking_pressure_features.py` | Ajout des caractéristiques de pression de *cracking* |
| `04_train_wazuh_native_loso.py` | Entraînement LightGBM et validation Leave-One-Scenario-Out (LOSO) |
| `model/` | Modèle entraîné (Wazuh-native, 248 caractéristiques) |

Les scripts sont numérotés dans leur **ordre d'exécution**.

## Prérequis

- Python 3.10+
- Le jeu de données **AIT-ADS** (Austrian Institute of Technology Alert Data Set) — non inclus dans le dépôt, à télécharger séparément
- Dépendances :

```bash
pip install lightgbm scikit-learn pandas numpy
```

## Reproduire le pipeline

Exécuter les scripts dans l'ordre :

```bash
python 00_inventory_wazuh_json.py
python 01_normalize_wazuh_json.py
python 02_apply_labels.py
python 03_build_wazuh_native_features.py
python 03b_add_rule_context_features.py
python 03c_add_cracking_pressure_features.py
python 04_train_wazuh_native_loso.py
```

Le dernier script entraîne le modèle et lance la validation LOSO : à chaque itération, un scénario AIT-ADS est entièrement retiré de l'entraînement et réservé au test.

## Résultats (validation LOSO)

| Métrique | Valeur moyenne |
|---|---|
| AUC | 0,97 |
| Recall@5 % | 0,943 |
| Lift@5 % | 18,9× |

En n'investiguant que les 5 % d'alertes les mieux notées, l'analyste retrouve en moyenne 94 % des attaques critiques.

## Méthodologie

- **Validation LOSO** plutôt qu'un découpage aléatoire, pour écarter toute fuite temporelle et mesurer la généralisation à des scénarios inconnus.
- **Rééquilibrage** appliqué au seul jeu d'entraînement ; le jeu de test conserve sa distribution naturelle.
- **Hyperparamètres fixés *a priori*** (pas de recherche sur grille sur le jeu de test), afin de ne pas biaiser l'évaluation.

## Avertissement

Code fourni à des fins de recherche et de reproductibilité dans un environnement de laboratoire contrôlé. Les performances ne sont pas directement transposables à un SOC de production sans validation complémentaire sur des données réelles.

## Auteur

Iheb Werfeli — École Hexagone, 2026.
