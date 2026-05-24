# Prédiction de la Gravité des Accidents de la Route (Données 2024)

## 1. Présentation du Projet
Ce projet est réalisé dans le cadre du module de **Data Science** et **Data Visualisation** de la **Licence 3 Science des Données et Numérique (SDN)** à la **FGES** (Faculté de Gestion, Économie et Sciences de l'Université Catholique de Lille).

**Membres de l'équipe :**
* Bensafia Chahine
* Noé Loisel
* Guehoudi Younes

**Enseignants Référents :** M. Almaksour (Data Science) & M. Demoli (Data Visualisation)

---

## 2. Consigne et Objectifs
Mandatés par une entreprise spécialisée en prévention des risques routiers, notre objectif est d'explorer les données des accidents de la circulation en France (2024), de comprendre les facteurs aggravants, et de développer un modèle prédictif de la gravité.

Le jeu de données `accidents_2024.csv` contient **54 402 accidents** et **14 variables**. La classe cible est fortement déséquilibrée (Seulement 7.3% de la modalité "Tué"). 

L'enjeu métier critique de ce projet est de maximiser le **Rappel (Recall) sur la classe "Tué"** afin de minimiser le nombre de faux négatifs (accidents mortels non détectés).

---

## 3. Architecture du Projet
Afin de professionnaliser le code, éviter le travail en silos et éliminer les redondances, l'architecture a été restructurée de manière modulaire :

    projetdata/
    ├── data/
    │   ├── raw/                 (Données brutes : accidents_2024.csv)
    │   └── processed/           (Données nettoyées et prêtes pour les modèles)
    ├── notebooks/               (Espace R&D)
    │   ├── 01_eda.ipynb         (Analyse exploratoire et Data Visualisation)
    │   └── 02_brouillons.ipynb  (Prototypes de modèles individuels)
    ├── src/                     (Code source de production)
    │   ├── preprocessing.py     (Nettoyage, imputation et One-Hot Encoding)
    │   ├── models.py            (Configuration des algorithmes)
    │   └── evaluation.py        (Validation croisée et calcul des métriques)
    ├── main.py                  (Script principal orchestrateur)
    ├── requirements.txt         (Dépendances du projet)
    └── README.md                (Ce fichier)

---

## 4. Installation

Clonez le dépôt sur votre machine locale :

    git clone https://github.com/younesguehhoudi/projetdata.git
    cd projetdata

Créez et activez un environnement virtuel (recommandé pour isoler les dépendances) :

    # Sous Windows :
    python -m venv venv
    venv\Scripts\activate

    # Sous Linux / macOS :
    python3 -m venv venv
    source venv/bin/activate

Installez les bibliothèques requises :

    pip install -r requirements.txt

---

## 5. Fonctionnement et Utilisation

L'intégralité du pipeline de Machine Learning est automatisée. Pour lancer le traitement des données, l'entraînement des 4 modèles (*Decision Tree, Régression Logistique, Random Forest, KNN*) et la comparaison des résultats, exécutez le script principal :

    python main.py

Le script se chargera de diviser les données de manière stratifiée, d'appliquer les paramètres de compensation du déséquilibre (class_weight='balanced') et d'évaluer les modèles via une validation croisée (5-fold) stricte.