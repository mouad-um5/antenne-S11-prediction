# =====================================================
# XGBOOST - ESTIMATION DE S11
# =====================================================

import pandas as pd
import numpy as np
import joblib # type: ignore

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# -----------------------------------------------------
# 1. Chargement des données
# -----------------------------------------------------

data = pd.read_csv(
    r"C:\Users\narji\OneDrive\Desktop\mm\pik_clean_ml.csv"
)

X = data[
    [
        "gap",
        "surface_width",
        "surface_length",
        "epsilon_r",
        "FREQUENCY"
    ]
]

y = data["s11"]

# -----------------------------------------------------
# 2. Train / Test
# -----------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------------------
# 3. Modèle XGBoost
# -----------------------------------------------------

xgb = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", xgb)
    ]
)

# -----------------------------------------------------
# 4. Entraînement
# -----------------------------------------------------

pipeline.fit(X_train, y_train)

# -----------------------------------------------------
# 5. Évaluation
# -----------------------------------------------------

y_pred = pipeline.predict(X_test)

print("\n===== PERFORMANCE DU MODELE =====")
print("MAE  :", mean_absolute_error(y_test, y_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2   :", r2_score(y_test, y_pred))

# -----------------------------------------------------
# 6. Sauvegarde du modèle
# -----------------------------------------------------

joblib.dump(pipeline, "s11_predictor.pkl")
print("\nModele sauvegarde : s11_predictor.pkl")

# -----------------------------------------------------
# 7. PREDICTION A PARTIR DES PARAMETRES
# -----------------------------------------------------

print("\n===== PREDICTION S11 =====")

gap = float(input("Gap : "))
width = float(input("Largeur substrat : "))
length = float(input("Longueur substrat : "))
epsilon = float(input("Epsilon_r : "))
freq = float(input("Frequence (GHz) : "))

X_new = pd.DataFrame(
    [[gap, width, length, epsilon, freq]],
    columns=[
        "gap",
        "surface_width",
        "surface_length",
        "epsilon_r",
        "FREQUENCY"
    ]
)

s11_pred = pipeline.predict(X_new)[0]

print(f"\nS11 estime = {s11_pred:.4f} dB")

# -----------------------------------------------------
# 8. Importance des variables (optionnel)
# -----------------------------------------------------

importances = pipeline.named_steps["model"].feature_importances_

plt.figure(figsize=(8, 4))
plt.barh(X.columns, importances)
plt.xlabel("Importance")
plt.title("Importance des parametres")
plt.tight_layout()
plt.show()
