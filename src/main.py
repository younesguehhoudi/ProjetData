import os
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Importation de nos propres modules
from src.preprocessing import preparer_donnees
from src.models import get_models

def main():
    BASE_DIR = Path(__file__).resolve().parent
    processed_dir = BASE_DIR / "data" / "processed"
    
    path_X_train = processed_dir / "X_train.csv"
    path_X_test = processed_dir / "X_test.csv"
    path_y_train = processed_dir / "y_train.csv"
    path_y_test = processed_dir / "y_test.csv"
    
    if not (path_X_train.exists() and path_X_test.exists() and path_y_train.exists() and path_y_test.exists()):
        print("Lancement du preprocessing automatique...")
        X_train, X_test, y_train, y_test = preparer_donnees(BASE_DIR)
    else:
        print("Fichiers de données récupérés depuis data/processed/.")
        X_train = pd.read_csv(path_X_train)
        X_test = pd.read_csv(path_X_test)
        y_train = pd.read_csv(path_y_train).squeeze()
        y_test = pd.read_csv(path_y_test).squeeze()

    models = get_models()
    resultats_synthese = []

    print(f"Évaluation sur {X_train.shape[1]} caractéristiques...")
    print("="*60)

    for name, config in models.items():
        print(f"Modèle : {name}...")
        model = config["model"]
        
        if config["scaled"]:
            scaler = StandardScaler()
            X_tr_final = scaler.fit_transform(X_train)
            X_te_final = scaler.transform(X_test)
        else:
            X_tr_final = X_train
            X_te_final = X_test
            
        model.fit(X_tr_final, y_train)
        y_pred = model.predict(X_te_final)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        accuracy = report["accuracy"]
        recall_tue = report.get("Tué", {}).get("recall", 0.0)
        
        resultats_synthese.append({"Modèle": name, "Accuracy": accuracy, "Recall (Tué)": recall_tue})
        print(f"-> Fait. Recall 'Tué': {recall_tue:.3f}")

    print("\n" + "="*60)
    print("TABLEAU DE SYNTHÈSE (Trié par Recall Tué)")
    print("="*60)
    df_res = pd.DataFrame(resultats_synthese).sort_values(by="Recall (Tué)", ascending=False)
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    main()