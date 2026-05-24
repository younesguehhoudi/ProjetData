import os
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


def preparer_donnees(base_dir):
    raw_dir       = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    figures_dir   = base_dir / "data" / "figures"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    path_brut = raw_dir / "accidents_2024.csv"

    for file_path in (base_dir / "data").glob("*.png"):
        print(f"Déplacement de {file_path.name} vers data/figures/...")
        os.rename(file_path, figures_dir / file_path.name)

    if not path_brut.exists():
        ancien_path = base_dir / "data" / "accidents_2024.csv"
        if ancien_path.exists():
            print("Déplacement du fichier source vers data/raw/...")
            os.rename(ancien_path, path_brut)
        else:
            raise FileNotFoundError(
                f"Fichier source introuvable. Placez 'accidents_2024.csv' dans : {raw_dir}"
            )

    print("Chargement des données brutes...")
    df = pd.read_csv(path_brut, sep=";", encoding="latin-1")

    df.drop_duplicates(inplace=True)
    df.dropna(subset=['Gravité (label)'], inplace=True)

    df['Gravité (label)'] = df['Gravité (label)'].replace({'O': '0_Indemne', '0': '0_Indemne'})

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    X = df.drop(columns=['Gravité (label)', 'Num_Acc', 'Département'])
    y = df['Gravité (label)']

    X_encoded = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    print("Sauvegarde des matrices dans data/processed/...")
    X_train.to_csv(processed_dir / "X_train.csv", index=False)
    X_test.to_csv(processed_dir / "X_test.csv", index=False)
    y_train.to_csv(processed_dir / "y_train.csv", index=False)
    y_test.to_csv(processed_dir / "y_test.csv", index=False)

    print("Preprocessing terminé.\n")
    return X_train, X_test, y_train, y_test