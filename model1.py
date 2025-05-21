# -*- coding: utf-8 -*-
"""
Created on Mon May 19 05:40:43 2025

@author: RAUL SANCHEZ
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# elegir dataset CSV

Tk().withdraw()  
file_path = askopenfilename(title="Selecciona tu archivo CSV", filetypes=[("CSV files", "*.csv")])

if not file_path:
    print("No seleccionaste ningún archivo. El programa terminará.")
    exit()

# crgar dataset
df = pd.read_csv(file_path)

print("\n--- Primeras 5 filas del dataset ---")
print(df.head())

print("\n--- Información general ---")
print(df.info())

print("\n--- Estadísticas descriptivas ---")
print(df.describe(include='all'))

# seleccion de columnas

print("\nColumnas disponibles:")
print(list(df.columns))

target_col = input("\nEscribe el nombre de la columna objetivo (variable a predecir): ")

if target_col not in df.columns:
    print("La columna objetivo no existe en el dataset. Termina el programa.")
    exit()

feature_cols = input("Escribe los nombres de las columnas características separadas por comas (ej: col1,col2,...): ")
features = [col.strip() for col in feature_cols.split(',')]

for col in features:
    if col not in df.columns:
        print(f"La columna {col} no existe. Termina el programa.")
        exit()

# limpiexa

df = df[[*features, target_col]].dropna()
print(f"\nDatos después de eliminar filas con valores faltantes: {len(df)} filas")

# preparar datos para ml

X = df[features]
y = df[target_col]

# Detectar si es clasificación o regresión según el tipo de la variable objetivo
if y.dtype == 'object' or len(y.unique()) <= 10:
    problem_type = 'classification'
    print("\nProblema detectado: Clasificación")
else:
    problem_type = 'regression'
    print("\nProblema detectado: Regresión")

# Convertir variables categóricas en dummies si existen en X
X = pd.get_dummies(X)

# Convertir variable objetivo para clasificación si es categórica
if problem_type == 'classification' and y.dtype == 'object':
    y = y.astype('category').cat.codes

# --- Paso 6: División train-test ---

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Paso 7: Crear y entrenar modelo ---

if problem_type == 'regression':
    model = LinearRegression()
else:
    model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# --- Paso 8: Predicción y evaluación ---

y_pred = model.predict(X_test)

if problem_type == 'regression':
    mse = mean_squared_error(y_test, y_pred)
    print(f"\nError cuadrático medio (MSE) en test: {mse:.4f}")

    # Graficar valores reales vs predichos
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel('Valores reales')
    plt.ylabel('Valores predichos')
    plt.title('Real vs Predicho')
    plt.show()

else:
    # Para clasificación convertir predicciones continuas a clases
    y_pred_class = (y_pred > 0.5).astype(int) if len(np.unique(y)) == 2 else model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_class)
    print(f"\nPrecisión (accuracy) en test: {acc:.4f}")

    # Matriz de confusión
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_test, y_pred_class)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
