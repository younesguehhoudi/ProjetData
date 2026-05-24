# Fichier : src/preprocessing.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def preparer_donnees(base_dir):
    """
    Orchestre le pipeline de traitement des données :
    Création des sous-dossiers (raw, processed, figures), nettoyage, encodage et découpage Train/Test.
    """
    # 1. Définition et création automatique des chemins cibles
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    figures_dir = base_dir / "data" / "figures"  # Dossier dédié aux graphiques
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    path_brut = raw_dir / "accidents_2024.csv"
    
    # Gestion des images volantes : si des fichiers .png sont directement dans data/, on les déplace dans figures/
    ancien_data_dir = base_dir / "data"
    for file_path in ancien_data_dir.glob("*.png"):
        print(f"Rangement de l'image {file_path.name} vers data/figures/...")
        os.rename(file_path, figures_dir / file_path.name)
    
    # Vérification de sécurité si le fichier initial n'est pas au bon endroit
    if not path_brut.exists():
        ancien_path = base_dir / "data" / "accidents_2024.csv"
        if ancien_path.exists():
            print("Déplacement du fichier original vers data/raw/...")
            os.rename(ancien_path, path_brut)
        else:
            raise FileNotFoundError(
                f"Le fichier source original est introuvable. Veuillez placer "
                f"'accidents_2024.csv' dans le dossier suivant : {raw_dir}"
            )

    print("Chargement des données brutes...")
    df = pd.read_csv(path_brut, sep=";", encoding="utf-8")
    
    # 2. Nettoyage et harmonisation
    if 'etat_surface' in df.columns:
        df['etat_surface'] = df['etat_surface'].replace('O', 'Normale')
        
    if 'vitesse_max' in df.columns:
        df['vitesse_max'] = df['vitesse_max'].astype(str).str.extract(r'(\d+)').astype(float)
        median_vitesse = df['vitesse_max'].median()
        df['vitesse_max'] = df['vitesse_max'].fillna(median_vitesse)
        df.loc[df['vitesse_max'] > 130, 'vitesse_max'] = median_vitesse
        
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    # 3. Encodage Universel (One-Hot Encoding)
    X = df.drop(columns=['Gravité (label)'])
    y = df['Gravité (label)']
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # 4. Split 80/20 Stratifié
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, 
        test_size=0.20, 
        stratify=y, 
        random_state=42
    )
    
    # 5. Sauvegarde dans le sous-dossier processed/
    print("Sauvegarde des matrices finalisées dans data/processed/...")
    X_train.to_csv(processed_dir / "X_train.csv", index=False)
    X_test.to_csv(processed_dir / "X_test.csv", index=False)
    y_train.to_csv(processed_dir / "y_train.csv", index=False)
    y_test.to_csv(processed_dir / "y_test.csv", index=False)
    
    print("Pipeline de preprocessing exécuté avec succès.\n")
    return X_train, X_test, y_train, y_test