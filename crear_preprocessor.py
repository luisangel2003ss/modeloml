import pandas as pd
import numpy as np
import re
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Cargar dataset
df = pd.read_csv("district_cleaned_ready_v2.csv")

def extraer_numero(texto):
    if pd.isna(texto):
        return np.nan
    match = re.search(r"[-+]?\d*\.\d+|\d+", str(texto))
    return float(match.group()) if match else np.nan

# Limpiar columnas objetivo
df["release_crude_oil"] = df["release_crude_oil"].apply(extraer_numero)
df["release_prod_wtr"] = df["release_prod_wtr"].apply(extraer_numero)
df = df.dropna(subset=["release_crude_oil", "release_prod_wtr"])

# Variables predictoras
X = df.drop(columns=["release_crude_oil", "release_prod_wtr"])
X["recovery_crude_oil_edit"] = df["recovery_crude_oil_edit"]
X["recovery_prod_water_edit"] = df["recovery_prod_water_edit"]

# Columnas para el preprocesador
columnas_numericas = X.select_dtypes(include=[np.number]).columns.tolist()
columnas_categoricas = X.select_dtypes(include=["object", "category"]).columns.tolist()

# Crear pipelines
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), columnas_numericas),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), columnas_categoricas)
])

# Entrenar preprocesador
preprocessor.fit(X)

# Guardar
joblib.dump(preprocessor, "preprocessor.pkl")
print("Preprocesador guardado con columnas consistentes al entrenamiento.")
