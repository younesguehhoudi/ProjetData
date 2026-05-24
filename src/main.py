import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# sorties de 01_preparation_donnees.ipynb : 103 features OHE, split 80/20 stratifié seed=42
X_train = pd.read_csv(os.path.join(BASE_DIR, '../data/X_train.csv'))
X_test  = pd.read_csv(os.path.join(BASE_DIR, '../data/X_test.csv'))
y_train = pd.read_csv(os.path.join(BASE_DIR, '../data/y_train.csv'))
y_test  = pd.read_csv(os.path.join(BASE_DIR, '../data/y_test.csv'))

y_train_1d = y_train.values.ravel()
y_test_1d  = y_test.values.ravel()

# jeu complet reconstruit uniquement pour la validation croisée
X_all = pd.concat([X_train, X_test])
y_all = pd.concat([y_train, y_test]).values.ravel()

# KNN et RL sont sensibles aux échelles — arbre et forêt ne le sont pas
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
X_all_scaled   = scaler.transform(X_all)

# class_weight='balanced' sur tous les modèles : Tué ne représente que 7% des cas
# Random Forest : poids x20 sur Tué — compromis recall/accuracy validé empiriquement
models = {
    "Random Forest (Noé)": {
        "model": RandomForestClassifier(
            n_estimators=500,
            class_weight={'Blessé léger': 1, 'Blessé hospitalisé': 2, 'Tué': 20},
            random_state=42
        ),
        "scaled": False
    },
    "Decision Tree (Younes)": {
        "model": DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42
        ),
        "scaled": False
    },
    "Régression Logistique (Chahine)": {
        "model": LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ),
        "scaled": True
    },
    "KNN": {
        "model": KNeighborsClassifier(n_neighbors=5),
        "scaled": True
    }
}

# métrique prioritaire : recall "Tué" — un faux négatif mortel coûte plus qu'une fausse alarme
resultats = []

for nom, config in models.items():
    print(f"\n{'='*60}")
    print(f"  {nom}")
    print(f"{'='*60}")

    m   = config["model"]
    use_scaled = config["scaled"]

    Xtr = X_train_scaled if use_scaled else X_train
    Xte = X_test_scaled  if use_scaled else X_test
    Xal = X_all_scaled   if use_scaled else X_all

    m.fit(Xtr, y_train_1d)
    y_pred = m.predict(Xte)

    acc = accuracy_score(y_test_1d, y_pred)
    print(f"\nAccuracy : {acc:.4f}")
    print(classification_report(y_test_1d, y_pred))
    print("Matrice de confusion :")
    print(confusion_matrix(y_test_1d, y_pred))

    # CV sur le jeu complet pour une estimation moins biaisée qu'un seul split
    scores = cross_val_score(m, Xal, y_all, cv=5, scoring='accuracy')
    print(f"\nValidation croisée (5-fold) : {scores.round(3)}")
    print(f"Moyenne CV : {scores.mean():.3f} (+/- {scores.std():.3f})")

    # recall "Tué" extrait séparément pour le tri du tableau comparatif
    from sklearn.metrics import recall_score
    classes = sorted(set(y_test_1d))
    recalls = recall_score(y_test_1d, y_pred, average=None, labels=classes)
    recall_tue = recalls[classes.index('Tué')] if 'Tué' in classes else 0

    resultats.append({
        "Modèle": nom,
        "Accuracy": round(acc, 3),
        "Recall Tué": round(recall_tue, 3),
        "CV Moyenne": round(scores.mean(), 3)
    })

# tri par recall "Tué" décroissant : critère de sélection du modèle retenu
print(f"\n{'='*60}")
print("  COMPARAISON FINALE DES 4 MODÈLES")
print(f"{'='*60}")
df_results = pd.DataFrame(resultats)
df_results = df_results.sort_values("Recall Tué", ascending=False)
print(df_results.to_string(index=False))
print(f"\n→ Métrique prioritaire : Recall Tué (minimiser les accidents mortels non détectés)")