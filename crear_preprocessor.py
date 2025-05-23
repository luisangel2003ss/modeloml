# -*- coding: utf-8 -*-
"""
Created on Wed May 21 15:01:05 2025

@author: RAUL SANCHEZ
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
import joblib

# Definir columnas
numeric_features = ['release_cond', 'release_gas']
categorical_features = ['probable_cause_edit', 'type_operation']

# Cargar dataset
df = pd.read_csv("district_cleaned_ready_v2_limpio.csv")

# Limpiar columnas numéricas si tienen unidades (por ejemplo: '2 GAL', '15 BBL')
for col in numeric_features:
    df[col] = df[col].astype(str).str.extract(r'(\d+\.?\d*)')  # Extrae solo el número
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convierte a float, fuerza NaN si falla

# Verifica si las columnas quedaron bien
print(df[numeric_features].dtypes)
print(df[numeric_features].head())

# Crear transformadores
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', RobustScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Crear preprocesador
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Entrenar preprocesador
X = df[numeric_features + categorical_features]
preprocessor.fit(X)

# Guardar preprocesador
joblib.dump(preprocessor, "preprocessor.pkl")

print("Preprocesador entrenado y guardado como 'preprocessor.pkl'")
