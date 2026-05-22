import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Chargement des données ---
X_train = pd.read_csv(os.path.join(BASE_DIR, '../data/X_train.csv'))
X_test = pd.read_csv(os.path.join(BASE_DIR, '../data/X_test.csv'))
y_train = pd.read_csv(os.path.join(BASE_DIR, '../data/y_train.csv'))
y_test = pd.read_csv(os.path.join(BASE_DIR, '../data/y_test.csv'))

# --- Modèle Random Forest ---
# class_weight manuel : pénalise davantage les erreurs sur "Tué" (classe rare, enjeu critique)
model = RandomForestClassifier(
    n_estimators=500,
    class_weight={'Blessé léger': 1, 'Blessé hospitalisé': 2, 'Tué': 20}
)
model.fit(X_train, y_train.values.ravel())
y_pred = model.predict(X_test)

# --- Métriques ---
print("Accuracy :", accuracy_score(y_test.values.ravel(), y_pred))
print(classification_report(y_test.values.ravel(), y_pred))
print("Matrice de confusion :\n", confusion_matrix(y_test.values.ravel(), y_pred))

# --- Feature importances (top 10) ---
importances = pd.Series(
    model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)
print("Top 10 features :\n", importances.head(10))