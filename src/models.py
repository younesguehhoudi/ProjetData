from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def get_models():
    return {
        "Random Forest": {
            "model": RandomForestClassifier(
                n_estimators=500,
                class_weight={'Blessé léger': 1, 'Blessé hospitalisé': 2, 'Tué': 20},
                random_state=42
            ),
            "scaled": False
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=42
            ),
            "scaled": False
        },
        "Régression Logistique": {
            "model": LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            ),
            "scaled": True
        },
        "KNN": {
            "model": KNeighborsClassifier(n_neighbors=80, weights='distance'),
            "scaled": True
        }
    }